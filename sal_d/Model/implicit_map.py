import torch
import numpy as np
from torch import nn

from sal_d.Model.layer_grad import LinearGrad, SoftplusGrad


class ImplicitMap(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        activation=None,
        latent_dropout=False,
        xyz_dim=3,
        geometric_init=True,
        beta=100,
    ):
        super().__init__()

        bias = 1.0
        self.latent_size = latent_size
        last_out_dim = 1
        dims = [latent_size + xyz_dim] + dims + [last_out_dim]
        self.d_in = latent_size + xyz_dim
        self.latent_in = latent_in
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = LinearGrad(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, -bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = SoftplusGrad(beta=beta)

    def forward(self, input, latent, compute_grad=False, cat_latent=True):
        """
        :param input: [shape: (N x d_in)]
        :param compute_grad: True for computing the input gradient. default=False
        :return: x: [shape: (N x d_out)]
                 x_grad: input gradient if compute_grad=True [shape: (N x d_in x d_out)]
                         None if compute_grad=False
        """

        x = input
        input_con = (
            latent.unsqueeze(1).repeat(1, input.shape[1], 1)
            if self.latent_size > 0
            else input
        )
        if self.latent_size > 0 and cat_latent:
            x = (
                torch.cat([x, input_con], dim=-1)
                if len(x.shape) == 3
                else torch.cat([x, latent.repeat(input.shape[0], 1)], dim=-1)
            )
        input_con = x
        to_cat = x
        x_grad = None

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.latent_in:
                x = torch.cat([x, to_cat], -1) / np.sqrt(2)
                if compute_grad:
                    skip_grad = (
                        torch.eye(self.d_in, device=x.device)[:, :3]
                        .unsqueeze(0)
                        .repeat(input.shape[0], input.shape[1], 1, 1)
                    )
                    x_grad = torch.cat([x_grad, skip_grad], 2) / np.sqrt(2)

            x, x_grad = lin(x, x_grad, compute_grad, l == 0)

            if l < self.num_layers - 2:
                x, x_grad = self.softplus(x, x_grad, compute_grad)

        return x, x_grad, input_con
