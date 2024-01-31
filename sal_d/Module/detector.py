import os
import json
import torch
import logging
import trimesh
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as offline
from tqdm import tqdm
from pyhocon import ConfigFactory
from plotly.subplots import make_subplots

from sal_d.Method.general import (
    get_class,
    get_cuda_ifavailable,
    mkdir_ifnotexists,
    as_mesh,
    compute_trimesh_chamfer,
)
from sal_d.Method.plots import plot_surface
from sal_d.Loss.sald import SALDLoss
from sal_d.Model.sald import SALD


def adjust_learning_rate(
    initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
):
    lr = initial_lr * ((1 / decreased_by) **
                       (num_iterations // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def optimize_latent(latent, ds, itemindex, network, lat_vecs):
    latent.detach_()
    latent.requires_grad_()
    lr = 1.0e-3
    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_func = SALDLoss(
        recon_loss_weight=1,
        grad_on_surface_weight=0,
        grad_loss_weight=0.1,
        z_weight=0.001,
        latent_reg_weight=0,
    )

    num_iterations = 800

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)
    network.with_sample = False
    network.adaptive_with_sample = False
    idx_latent = get_cuda_ifavailable(torch.arange(lat_vecs.num_embeddings))
    for e in range(num_iterations):
        # network.with_sample = e > 100
        pnts_mnfld, normals_mnfld, sample_nonmnfld, indices = ds[itemindex]

        pnts_mnfld = get_cuda_ifavailable(pnts_mnfld).unsqueeze(0)
        normals_mnfld = get_cuda_ifavailable(normals_mnfld).unsqueeze(0)
        sample_nonmnfld = get_cuda_ifavailable(sample_nonmnfld).unsqueeze(0)
        indices = get_cuda_ifavailable(indices).unsqueeze(0)
        outside_latent = lat_vecs(
            idx_latent[np.random.choice(
                np.arange(lat_vecs.num_embeddings), 4, False)]
        )

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        outputs = network(
            pnts_mnfld,
            None,
            sample_nonmnfld[:, :, :3],
            latent,
            False,
            only_decoder_forward=False,
        )
        loss_res = loss_func(
            network_outputs=outputs,
            normals_gt=normals_mnfld,
            normals_nonmnfld_gt=sample_nonmnfld[:, :, 3:6],
            pnts_mnfld=pnts_mnfld,
            gt_nonmnfld=sample_nonmnfld[:, :, -1],
            epoch=-1,
        )
        loss = loss_res["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info("iteration : {0} , loss {1}".format(e, loss.item()))
        logging.info(
            "mean {0} , std {1}".format(
                latent.mean().item(), latent.std().item())
        )

    # network.with_sample = True
    return latent


class Detector(object):
    def __init__(self) -> None:
        self.model_folder_path = "./pretrained/"
        self.device = "cpu"
        return

    def evaluate_with_load(
        self,
        conf,
        exps_folder_name,
        override,
        resolution,
        recon_only=False,
        plot_cmpr=True,
        with_opt=False,
    ):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.info("running")
        conf = ConfigFactory.parse_file(conf)

        if override != "":
            expname = override
        else:
            expname = conf.get_string("train.expname")

        saved_model_state = torch.load(
            self.model_folder_path + "ModelParameters/latest.pth",
            map_location=torch.device(self.device),
        )
        logging.info("loaded model")
        saved_model_epoch = saved_model_state["epoch"]

        network = SALD(latent_size=256)

        network.load_state_dict(
            {
                ".".join(k.split(".")[1:]): v
                for k, v in saved_model_state["model_state_dict"].items()
            }
        )

        self.evaluate(
            network=get_cuda_ifavailable(network),
            exps_folder_name=exps_folder_name,
            experiment_name=expname,
            epoch=saved_model_epoch,
            resolution=resolution,
            conf=conf,
            index=-1,
            recon_only=recon_only,
            plot_cmpr=plot_cmpr,
            with_gt=True,
            with_opt=with_opt,
        )

    def evaluate(
        self,
        network,
        exps_folder_name,
        experiment_name,
        epoch,
        resolution,
        conf,
        index,
        recon_only,
        plot_cmpr=False,
        with_gt=False,
    ):
        if isinstance(network, torch.nn.parallel.DataParallel):
            network = network.module

        chamfer_results = dict(
            files=[],
            reg_to_gen_chamfer=[],
            gen_to_reg_chamfer=[],
            scan_to_gen_chamfer=[],
            gen_to_scan_chamfer=[],
        )

        split_filename = "./confs/splits/{0}".format(
            conf.get_string("train.test_split")
        )
        with open(split_filename, "r") as f:
            split = json.load(f)

        ds = get_class(conf.get_string("train.dataset.class"))(
            split=split,
            with_gt=with_gt,
            with_scans=True,
            scans_file_type="ply",
            **conf.get_config("train.dataset.properties"),
        )
        total_files = len(ds)
        logging.info("total files : {0}".format(total_files))
        prop = conf.get_config("train.dataset.properties")
        prop["number_of_points"] = int(np.sqrt(30000))
        ds_eval_scan = get_class(conf.get_string("train.dataset.class"))(
            split=split, with_gt=True, **prop
        )

        mkdir_ifnotexists(
            os.path.join(
                conf.get_string("train.base_path"),
                exps_folder_name,
                experiment_name,
                "evaluation",
            )
        )
        mkdir_ifnotexists(
            os.path.join(
                conf.get_string("train.base_path"),
                exps_folder_name,
                experiment_name,
                "evaluation",
                split_filename.split("/")[-1].split(".json")[0],
            )
        )
        path = os.path.join(
            conf.get_string("train.base_path"),
            exps_folder_name,
            experiment_name,
            "evaluation",
            split_filename.split("/")[-1].split(".json")[0],
            str(epoch),
        )

        mkdir_ifnotexists(path)

        counter = 0
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )

        names = [
            "_".join(
                [
                    ds.npyfiles_dist[i].split("/")[-3:][0],
                    ds.npyfiles_dist[i].split("/")[-3:][2],
                ]
            ).split("_dist_triangle.npy")[0]
            for i in range(len(ds.npyfiles_dist))
        ]

        i = 1
        # index = index + 1
        for data in tqdm(dataloader):
            if index == -1 or index == i:
                logging.info(counter)
                # logging.info (ds.npyfiles_mnfld[data[-1].item()].split('/'))
                counter = counter + 1

                [
                    logging.debug("evaluating " +
                                  ds.npyfiles_mnfld[data[-1][i]])
                    for i in range(len(data[-1]))
                ]

                input_pc = get_cuda_ifavailable(data[0])
                input_normal = get_cuda_ifavailable(data[1])
                filename = [
                    "{0}/nonuniform_iteration_{1}_{2}_id.ply".format(
                        path,
                        epoch,
                        ds.npyfiles_mnfld[data[-1][i].item()].split("/")[-3]
                        + "_"
                        + ds.npyfiles_mnfld[data[-1][i].item()]
                        .split("/")[-1]
                        .split(".npy")[0],
                    )
                    for i in range(len(data[-1]))
                ][0]

                _, latent, _ = network(
                    manifold_points=input_pc,
                    manifold_normals=input_normal,
                    sample_nonmnfld=None,
                    latent=None,
                    only_encoder_forward=True,
                    only_decoder_forward=False,
                )
                pnts_to_plot = input_pc

                if os.path.isfile(filename):
                    reconstruction = trimesh.load(filename)
                    logging.info("loaded : {0}".format(filename))
                else:
                    if latent is not None:
                        reconstruction, _ = plot_surface(
                            with_points=False,
                            points=pnts_to_plot.detach()[0],
                            decoder=network,
                            latent=latent,
                            path=path,
                            epoch=epoch,
                            in_epoch=ds.npyfiles_mnfld[data[-1].item()
                                                       ].split("/")[-3]
                            + "_"
                            + ds.npyfiles_mnfld[data[-1].item()]
                            .split("/")[-1]
                            .split(".npy")[0],
                            shapefile=ds.npyfiles_mnfld[data[-1].item()],
                            resolution=resolution,
                            mc_value=0,
                            is_uniform_grid=False,
                            verbose=True,
                            save_html=False,
                            save_ply=True,
                            overwrite=True,
                            is_3d=True,
                            z_func={"id": lambda x: x},
                        )
                if reconstruction is None and latent is not None:
                    i = i + 1
                    continue

                if not recon_only:
                    normalization_params_filename = ds.normalization_files[data[-1]]
                    logging.debug(
                        "normalization params are " + normalization_params_filename
                    )

                    normalization_params = np.load(
                        normalization_params_filename, allow_pickle=True
                    )
                    scale = normalization_params.item()["scale"]
                    center = normalization_params.item()["center"]

                    if with_gt:
                        gt_mesh_filename = ds.gt_files[data[-1]]
                        ground_truth_points = trimesh.Trimesh(
                            trimesh.sample.sample_surface(
                                as_mesh(trimesh.load(gt_mesh_filename)), 30000
                            )[0]
                        )
                        dists_to_reg = compute_trimesh_chamfer(
                            gt_points=ground_truth_points,
                            gen_mesh=reconstruction,
                            offset=-center,
                            scale=1.0 / scale,
                        )

                    dists_to_scan = compute_trimesh_chamfer(
                        gt_points=trimesh.Trimesh(
                            ds_eval_scan[data[-1]][0].cpu().numpy()
                        ),
                        gen_mesh=reconstruction,
                        offset=0,
                        scale=1.0,
                    )

                    if with_gt:
                        chamfer_results["files"].append(ds.gt_files[data[-1]])
                        chamfer_results["reg_to_gen_chamfer"].append(
                            dists_to_reg["gt_to_gen_chamfer"]
                        )
                        chamfer_results["gen_to_reg_chamfer"].append(
                            dists_to_reg["gen_to_gt_chamfer"]
                        )

                    chamfer_results["scan_to_gen_chamfer"].append(
                        dists_to_scan["gt_to_gen_chamfer"]
                    )
                    chamfer_results["gen_to_scan_chamfer"].append(
                        dists_to_scan["gen_to_gt_chamfer"]
                    )

                    if plot_cmpr:
                        fig = make_subplots(
                            rows=1,
                            cols=2 + int(with_gt),
                            specs=[[{"type": "scene"}] * (2 + int(with_gt))],
                            subplot_titles=(
                                "input scan", "Ours", "Registration")
                            if with_gt
                            else ("input pc", "Ours"),
                        )

                        fig.layout.scene.update(
                            dict(
                                camera=dict(
                                    up=dict(x=0, y=1, z=0),
                                    center=dict(x=0, y=0.0, z=0),
                                    eye=dict(x=0, y=0.6, z=0.9),
                                ),
                                xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                yaxis=dict(range=[-1.5, 1.5], autorange=False),
                                zaxis=dict(range=[-1.5, 1.5], autorange=False),
                                aspectratio=dict(x=1, y=1, z=1),
                            )
                        )
                        fig.layout.scene2.update(
                            dict(
                                camera=dict(
                                    up=dict(x=0, y=1, z=0),
                                    center=dict(x=0, y=0.0, z=0),
                                    eye=dict(x=0, y=0.6, z=0.9),
                                ),
                                xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                yaxis=dict(range=[-1.5, 1.5], autorange=False),
                                zaxis=dict(range=[-1.5, 1.5], autorange=False),
                                aspectratio=dict(x=1, y=1, z=1),
                            )
                        )
                        if with_gt:
                            fig.layout.scene3.update(
                                dict(
                                    camera=dict(
                                        up=dict(x=0, y=1, z=0),
                                        center=dict(x=0, y=0.0, z=0),
                                        eye=dict(x=0, y=0.6, z=0.9),
                                    ),
                                    xaxis=dict(
                                        range=[-1.5, 1.5], autorange=False),
                                    yaxis=dict(
                                        range=[-1.5, 1.5], autorange=False),
                                    zaxis=dict(
                                        range=[-1.5, 1.5], autorange=False),
                                    aspectratio=dict(x=1, y=1, z=1),
                                )
                            )

                        scan_mesh = as_mesh(
                            trimesh.load(ds.scans_files[data[-1]]))

                        scan_mesh.vertices = (
                            scan_mesh.vertices - center) / scale

                        def tri_indices(simplices):
                            return (
                                [triplet[c] for triplet in simplices] for c in range(3)
                            )

                        I, J, K = tri_indices(scan_mesh.faces)
                        color = "#ffffff"
                        trace = go.Mesh3d(
                            x=scan_mesh.vertices[:, 0],
                            y=scan_mesh.vertices[:, 1],
                            z=scan_mesh.vertices[:, 2],
                            i=I,
                            j=J,
                            k=K,
                            name="scan",
                            color=color,
                            opacity=1.0,
                            flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1),
                        )
                        fig.add_trace(trace, row=1, col=1)

                        I, J, K = tri_indices(reconstruction.faces)
                        color = "#ffffff"
                        trace = go.Mesh3d(
                            x=reconstruction.vertices[:, 0],
                            y=reconstruction.vertices[:, 1],
                            z=reconstruction.vertices[:, 2],
                            i=I,
                            j=J,
                            k=K,
                            name="our",
                            color=color,
                            opacity=1.0,
                            flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1),
                        )

                        fig.add_trace(trace, row=1, col=2)

                        if with_gt:
                            gtmesh = as_mesh(trimesh.load(gt_mesh_filename))
                            gtmesh.vertices = (
                                gtmesh.vertices - center) / scale
                            I, J, K = tri_indices(gtmesh.faces)
                            trace = go.Mesh3d(
                                x=gtmesh.vertices[:, 0],
                                y=gtmesh.vertices[:, 1],
                                z=gtmesh.vertices[:, 2],
                                i=I,
                                j=J,
                                k=K,
                                name="gt",
                                color=color,
                                opacity=1.0,
                                flatshading=False,
                                lighting=dict(
                                    diffuse=1, ambient=0, specular=0),
                                lightposition=dict(x=0, y=0, z=-1),
                            )

                            fig.add_trace(trace, row=1, col=3)

                        div = offline.plot(
                            fig,
                            include_plotlyjs=False,
                            output_type="div",
                            auto_open=False,
                        )
                        div_id = (
                            div.split("=")[1]
                            .split()[0]
                            .replace("'", "")
                            .replace('"', "")
                        )

                        js = """
                                        <script>
                                        var gd = document.getElementById('{div_id}');
                                        var isUnderRelayout = false

                                        gd.on('plotly_relayout', () => {{
                                        console.log('relayout', isUnderRelayout)
                                        if (!isUnderRelayout) {{
                                                Plotly.relayout(
                                                    gd, 'scene2.camera', gd.layout.scene.camera)
                                                .then(() => {{ isUnderRelayout = false }}  )
                                                Plotly.relayout(
                                                    gd, 'scene3.camera', gd.layout.scene.camera)
                                                .then(() => {{ isUnderRelayout = false }}  )
                                            }}

                                        isUnderRelayout = true;
                                        }})
                                        </script>""".format(div_id=div_id)

                        # merge everything
                        div = (
                            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
                            + div
                            + js
                        )
                        with open(
                            os.path.join(
                                path,
                                "compare_{0}.html".format(
                                    ds.npyfiles_mnfld[data[-1]
                                                      [0].item()].split("/")[-3]
                                    + "_"
                                    + ds.npyfiles_mnfld[data[-1][0].item()]
                                    .split("/")[-1]
                                    .split(".npy")[0]
                                ),
                            ),
                            "w",
                        ) as text_file:
                            text_file.write(div)
            i = i + 1
            logging.info(i)
        if index == -1:
            pd.DataFrame(chamfer_results).to_csv(
                os.path.join(path, "eval_results.csv"))
            # with open(os.path.join(path,"chamfer.csv"),"w",) as f:
            #     if (with_opt):
            #         f.write("shape, chamfer_dist, chamfer scan dist, after opt chamfer dist, after opt chamfer scan dist\n")
            #         for result in chamfer_results:
            #             f.write("{}, {} , {}\n".format(result[0], result[1], result[2], result[3], result[4]))
            #     else:
            #         f.write("shape, chamfer_dist, chamfer scan dist\n")
            #         for result in chamfer_results:
            #             f.write("{}, {} , {}\n".format(result[0], result[1], result[2]))

    def detect(self) -> bool:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument(
            "--expname",
            required=False,
            help="The experiment name to be evaluated.",
            default="",
        )
        arg_parser.add_argument(
            "--override", required=False, help="Override exp name.", default=""
        )
        arg_parser.add_argument(
            "--exps_folder_name", default="exps", help="The experiments directory."
        )
        arg_parser.add_argument(
            "--split", required=False, help="The split to evaluate.", default=""
        )
        arg_parser.add_argument(
            "--with_opt",
            default=False,
            action="store_true",
            help="If set, optimizing latent with reconstruction Loss versus input scan",
        )
        arg_parser.add_argument(
            "--resolution", default=256, type=int, help="Grid resolution"
        )
        arg_parser.add_argument("--index", default=-1, type=int, help="")
        arg_parser.add_argument(
            "--recon_only", default=False, action="store_true")
        arg_parser.add_argument(
            "--plot_cmpr", default=False, action="store_true")

        logger = logging.getLogger()

        logger.setLevel(logging.DEBUG)
        logging.info("running")

        args = arg_parser.parse_args()

        conf = "./sal_d/Config/dfaust_vae.conf"

        self.evaluate_with_load(
            conf=conf,
            exps_folder_name=args.exps_folder_name,
            resolution=args.resolution,
            override=args.override,
            with_opt=args.with_opt,
        )
        return True
