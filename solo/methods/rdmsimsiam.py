# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from solo.utils.misc import gather
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simsiam import simsiam_loss_func
from solo.methods.base import BaseMethod


class RDMSimSiam(BaseMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        sigma: float =1,
        pred_type:str = "poly",
        pred_location:str = "online",
        **kwargs,
    ):
        """Implements RDMSimSiam.

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
        """

        super().__init__(**kwargs)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        self.sigma = sigma
        self.pred_type = pred_type
        self.pred_location = pred_location


    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(RDMSimSiam, RDMSimSiam).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("rdmsimsiam")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--sigma", type=float, default=1)
        parser.add_argument("--pred_type", type=str, default="poly")
        parser.add_argument("--pred_location", type=str, default="online")
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params: List[dict] = [
            {"params": self.projector.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimSiam reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimSiam loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        # ------- contrastive loss -------
        neg_cos_sim = self.loss(z1,z2)/2 +self.loss(z2,z1)/2

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss


    def loss(self,p, z):
        z, p = gather(z), gather(p)
        if self.pred_location == "online": 
            with torch.no_grad():
                u, s, vh = torch.linalg.svd(p.type(torch.cuda.FloatTensor), full_matrices=False)

                if self.pred_type == "poly":
                    w = vh.T @ torch.diag_embed(s.pow(self.sigma)) @ vh
                elif self.pred_type == "log":
                    w = vh.T @ torch.diag_embed(s.log()) @ vh
                elif self.pred_type == "log_1":
                    w = vh.T @ torch.diag_embed((s+1).log()) @ vh
                elif self.pred_type == "log_2":
                    w = vh.T @ torch.diag_embed((s.pow(2)+1).log()) @ vh
                w = w.type(torch.cuda.HalfTensor)

            p = p @ w.detach() 
            return simsiam_loss_func(p,z)

        elif self.pred_location == "target":
            with torch.no_grad():
                u, s, vh = torch.linalg.svd(z.type(torch.cuda.FloatTensor), full_matrices=False)
                z = (u @ torch.diag_embed(s.pow(1+self.sigma)) @ vh).type(torch.cuda.HalfTensor)
            return simsiam_loss_func(p,z)