import torch
import logging
from torch import nn
from torch import distributions as dist

from sal_d.Model.simple_pointnet import SimplePointnet
from sal_d.Model.implicit_map import ImplicitMap


class SALD(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.latent_size = latent_size
        self.with_normals = False
        encoder_input_size = 6 if self.with_normals else 3

        self.encoder = SimplePointnet(
            hidden_dim=2 * latent_size, c_dim=latent_size, dim=encoder_input_size
        )

        self.implicit_map = ImplicitMap(
            latent_size=latent_size,
            dims=[512, 512, 512, 512, 512, 512, 512, 512],
            dropout=[],
            dropout_prob=0.2,
            norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
            latent_in=[4],
            activation=None,
            latent_dropout=False,
            weight_norm=True,
            xyz_dim=3,
        )

        self.predict_normals_on_surfce = True

        logging.debug(
            """self.latent_size = {0},
                      self.with_normals = {1}
                      self.predict_normals_on_surfce = {2}
                      """.format(
                self.latent_size, self.with_normals, self.predict_normals_on_surfce
            )
        )

    def forward(
        self,
        manifold_points,
        manifold_normals,
        sample_nonmnfld,
        latent,
        only_encoder_forward,
        only_decoder_forward,
        epoch=-1,
    ):
        output = {}

        if self.encoder is not None and not only_decoder_forward:
            encoder_input = (
                torch.cat([manifold_points, manifold_normals], axis=-1)
                if self.with_normals
                else manifold_points
            )
            q_latent_mean, q_latent_std = self.encoder(encoder_input)

            q_z = dist.Normal(q_latent_mean, torch.exp(q_latent_std))
            latent = q_z.rsample()
            latent_reg = q_latent_mean.abs().mean(dim=-1) + (
                q_latent_std + 1
            ).abs().mean(dim=-1)
            output["latent_reg"] = latent_reg

            if only_encoder_forward:
                return latent, q_latent_mean, torch.exp(q_latent_std)
        else:
            if only_encoder_forward:
                return None, None, None

        if only_decoder_forward:
            return self.implicit_map(manifold_points, latent, False)[0]
        else:
            non_mnfld_pred, non_mnfld_pred_grad, _ = self.implicit_map(
                sample_nonmnfld, latent, True
            )

            output["non_mnfld_pred_grad"] = non_mnfld_pred_grad
            output["non_mnfld_pred"] = non_mnfld_pred

            if not latent is None:
                output["norm_square_latent"] = (latent**2).mean(-1)

            if self.predict_normals_on_surfce:
                _, grad_on_surface, _ = self.implicit_map(
                    manifold_points, latent, True)
                output["grad_on_surface"] = grad_on_surface

            return output
