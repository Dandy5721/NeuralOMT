import torch
from torch.nn import Module

# from geomloss.sinkhorn_samples import sinkhorn_tensorized
from .sinkhorn_samples import sinkhorn_tensorized
from geomloss.sinkhorn_samples import sinkhorn_online
from geomloss.sinkhorn_samples import sinkhorn_multiscale

from geomloss.kernel_samples import kernel_tensorized as hausdorff_tensorized
from geomloss.kernel_samples import kernel_online as hausdorff_online
from geomloss.kernel_samples import kernel_multiscale as hausdorff_multiscale
from geomloss.kernel_samples import kernel_tensorized, kernel_online, kernel_multiscale
from functools import partial
import warnings

routines = {
    "sinkhorn": {
        "tensorized": sinkhorn_tensorized,
        "online": sinkhorn_online,
        "multiscale": sinkhorn_multiscale,
    },
    "hausdorff": {
        "tensorized": hausdorff_tensorized,
        "online": hausdorff_online,
        "multiscale": hausdorff_multiscale,
    },
    "energy": {
        "tensorized": partial(kernel_tensorized, name="energy"),
        "online": partial(kernel_online, name="energy"),
        "multiscale": partial(kernel_multiscale, name="energy"),
    },
    "gaussian": {
        "tensorized": partial(kernel_tensorized, name="gaussian"),
        "online": partial(kernel_online, name="gaussian"),
        "multiscale": partial(kernel_multiscale, name="gaussian"),
    },
    "laplacian": {
        "tensorized": partial(kernel_tensorized, name="laplacian"),
        "online": partial(kernel_online, name="laplacian"),
        "multiscale": partial(kernel_multiscale, name="laplacian"),
    },
}

def exp_single(x):
    return torch.exp(-x)

class euclidean_distance_dwd_loss(Module):
    def __init__(
            self,
            loss="sinkhorn",
            p=2,
            blur=0.05,
            reach=None,
            diameter=None,
            scaling=0.5,
            truncate=5,
            cost=None,
            kernel=None,
            cluster_scale=None,
            debias=True,
            potentials=False,
            verbose=False,
            backend="auto",
            size=10,
            k = 1.0
    ):

        super(euclidean_distance_dwd_loss, self).__init__()
        self.loss = loss
        self.backend = backend
        self.p = p
        self.blur = blur
        self.reach = reach
        self.truncate = truncate
        self.diameter = diameter
        self.scaling = scaling
        self.cost = cost
        self.kernel = kernel
        self.cluster_scale = cluster_scale
        self.debias = debias
        self.potentials = potentials
        self.verbose = verbose
        self.size = size
        self.k = k

    def forward(self, distance, adj, *args, extra_value=True):
        """Computes the loss between sampled measures.

        Documentation and examples: Soon!
        Until then, please check the tutorials :-)"""
        # print('.....', len(args), ".....")
        if len(args) != 2:
            raise ValueError(
                "A euclidean_distance_wd_loss accepts two (x, y), four (α, x, β, y) or six (l_x, α, x, l_y, β, y)  arguments."
            )
        res = []
        pred, true = args  # (N,1)
        infeat_num = pred.shape[0]
        if pred.dim() != 2:
            raise ValueError(
                "input dim must be 2 dimention"

            )
        all_num = self.get_valid_node_num(pred)
        mask = (distance !=0).float()  #need to recover
        zeros_m = torch.zeros((1, 1)).to(pred.device)

        exp_sum_dis = (exp_single(distance[:, :] * mask)).sum(1).unsqueeze(1)  # b,n -> b
        exp_sum_dis = torch.cat((exp_sum_dis, zeros_m), dim=0)
        pred = torch.cat((pred, zeros_m), dim=0)
        true = torch.cat((true, zeros_m), dim=0)
        if extra_value:
            # infeat_num -= 1
            mask[:-1, 0] = 1
        else:
            mask[:, 0] = 1
        for i in range(0, infeat_num, self.size):
            tmp = adj[i:i + self.size, :].long()  # distance self -> 1
            tmp_mask = mask[i:i + self.size, :]
            tmp_dis = exp_single(distance[i:i + self.size, :]).unsqueeze(2) / (exp_sum_dis[tmp, :] + 1e-6)
            tmp_dis[:, 0, :] = 1.0
            #             # print(pred[tmp, :],"----")
            # pred_, true_ = pred[tmp, :] * tmp_dis, true[tmp, :] * tmp_dis  # a, b,n   need to recover
                        # print("----", pred_, pred_.shape)
            pred_, true_ = pred[tmp, :], true[tmp, :]
            pred_[:, 1:, :] = pred_[:, 1:, :].detach()

            # pred_, true_ = pred[tmp, :], true[tmp, :]
            distance_ = distance[i:i + self.size, :]
            # --------------------------------------------------------------
            l_x, α, x, l_y, β, y = self.process_args(pred_, true_, all_num)
            B, N, M, D, l_x, α, l_y, β = self.check_shapes(l_x, α, x, l_y, β, y)
            #             print("--------", x)
            backend = (
                self.backend
            )  # Choose the backend -----------------------------------------
            if l_x is not None or l_y is not None:
                if backend in ["auto", "multiscale"]:
                    backend = "multiscale"
                else:
                    raise ValueError(
                        'Explicit cluster labels are only supported with the "auto" and "multiscale" backends.'
                    )

            elif backend == "auto":
                if M * N <= 5000 ** 2:
                    backend = (
                        "tensorized"  # Fast backend, with a quadratic memory footprint
                    )
                else:
                    if (
                            D <= 3
                            and self.loss == "sinkhorn"
                            and M * N > 10000 ** 2
                            and self.p == 2
                    ):
                        backend = "multiscale"  # Super scalable algorithm in low dimension
                    else:
                        backend = "online"  # Play it safe, without kernel truncation

            # Check compatibility between the batchsize and the backend --------------------------

            if backend in ["multiscale"]:  # multiscale routines work on single measures
                if B == 1:
                    α, x, β, y = α.squeeze(0), x.squeeze(0), β.squeeze(0), y.squeeze(0)
                elif B > 1:
                    warnings.warn(
                        "The 'multiscale' backend do not support batchsize > 1. "
                        + "Using 'tensorized' instead: beware of memory overflows!"
                    )
                    backend = "tensorized"

            if B == 0 and backend in [
                "tensorized",
                "online",
            ]:  # tensorized and online routines work on batched tensors
                α, x, β, y = α.unsqueeze(0), x.unsqueeze(0), β.unsqueeze(0), y.unsqueeze(0)

            # Run --------------------------------------------------------------------------------
            values = routines[self.loss][backend](
                α,
                x,
                β,
                y,
                distance_,
                tmp_mask,
                k = self.k,
                use_dis = True,
                p=self.p,
                blur=self.blur,
                reach=self.reach,
                diameter=self.diameter,
                scaling=self.scaling,
                truncate=self.truncate,
                cost=self.cost,
                kernel=self.kernel,
                cluster_scale=self.cluster_scale,
                debias=self.debias,
                potentials=self.potentials,
                labels_x=l_x,
                labels_y=l_y,
                verbose=self.verbose,
            )

            # Make sure that the output has the correct shape ------------------------------------
            #             if (
            #                 self.potentials
            #             ):  # Return some dual potentials (= test functions) sampled on the input measures
            #                 F, G = values
            #                 return F.view_as(α), G.view_as(β)

            #             else:  # Return a scalar cost value
            #                 if backend in ["multiscale"]:  # KeOps backends return a single scalar value
            #                     return values.view(-1)  # The user expects a "batch list" of distances

            #                 else:  # "tensorized" backend returns a "batch vector" of values
            #                     return values  # The user expects a "batch vector" of distances
            # --------------------------------------------------------------
            res.append(values)
        res = torch.concat(res, dim=0)
        return res.sum()

    def get_valid_node_num(self, predict, eps=1e-6):
        B = predict.shape[0]
        # mask = (distance != 0)
        # count_num_matrix = torch.sum(mask.view(len(distance), -1), dim=1)
        # zeros_node = (count_num_matrix == 0)
        # res_num = B - zeros_node.sum().float().item() + eps
        return B

    def process_args(self, x, y, n):
        α = self.generate_weights(x, n)
        β = α
        # print("α", α)
        return None, α, x, None, β, y

    def generate_weights(self, x, n):
        if x.dim() == 3:
            B, N, _ = x.shape
            tmp = torch.ones(B, N).type_as(x)
            return tmp / n
        else:
            raise ValueError(
                "Input samples 'x' and 'y' should be encoded as (B,N,D) (batch) tensors."
            )

    def check_shapes(self, l_x, α, x, l_y, β, y):

        if α.dim() != β.dim():
            raise ValueError(
                "Input weights 'α' and 'β' should have the same number of dimensions."
            )
        if x.dim() != y.dim():
            raise ValueError(
                "Input samples 'x' and 'y' should have the same number of dimensions."
            )
        if x.shape[-1] != y.shape[-1]:
            raise ValueError(
                "Input samples 'x' and 'y' should have the same last dimension."
            )

        if (
                x.dim() == 2
        ):  # No batch --------------------------------------------------------------------
            B = 0  # Batchsize
            N, D = x.shape  # Number of "i" samples, dimension of the feature space
            M, _ = y.shape  # Number of "j" samples, dimension of the feature space

            if α.dim() not in [1, 2]:
                raise ValueError(
                    "Without batches, input weights 'α' and 'β' should be encoded as (N,) or (N,1) tensors."
                )
            elif α.dim() == 2:
                if α.shape[1] > 1:
                    raise ValueError(
                        "Without batches, input weights 'α' should be encoded as (N,) or (N,1) tensors."
                    )
                if β.shape[1] > 1:
                    raise ValueError(
                        "Without batches, input weights 'β' should be encoded as (M,) or (M,1) tensors."
                    )
                α, β = α.view(-1), β.view(-1)

            if l_x is not None:
                if l_x.dim() not in [1, 2]:
                    raise ValueError(
                        "Without batches, the vector of labels 'l_x' should be encoded as an (N,) or (N,1) tensor."
                    )
                elif l_x.dim() == 2:
                    if l_x.shape[1] > 1:
                        raise ValueError(
                            "Without batches, the vector of labels 'l_x' should be encoded as (N,) or (N,1) tensors."
                        )
                    l_x = l_x.view(-1)
                if len(l_x) != N:
                    raise ValueError(
                        "The vector of labels 'l_x' should have the same length as the point cloud 'x'."
                    )

            if l_y is not None:
                if l_y.dim() not in [1, 2]:
                    raise ValueError(
                        "Without batches, the vector of labels 'l_y' should be encoded as an (M,) or (M,1) tensor."
                    )
                elif l_y.dim() == 2:
                    if l_y.shape[1] > 1:
                        raise ValueError(
                            "Without batches, the vector of labels 'l_y' should be encoded as (M,) or (M,1) tensors."
                        )
                    l_y = l_y.view(-1)
                if len(l_y) != M:
                    raise ValueError(
                        "The vector of labels 'l_y' should have the same length as the point cloud 'y'."
                    )

            N2, M2 = α.shape[0], β.shape[0]

        elif (
                x.dim() == 3
        ):  # batch computation ---------------------------------------------------------
            (
                B,
                N,
                D,
            ) = x.shape
            # Batchsize, number of "i" samples, dimension of the feature space
            (
                B2,
                M,
                _,
            ) = y.shape
            # Batchsize, number of "j" samples, dimension of the feature space
            if B != B2:
                raise ValueError("Samples 'x' and 'y' should have the same batchsize.")

            if α.dim() not in [2, 3]:
                raise ValueError(
                    "With batches, input weights 'α' and 'β' should be encoded as (B,N) or (B,N,1) tensors."
                )
            elif α.dim() == 3:
                if α.shape[2] > 1:
                    raise ValueError(
                        "With batches, input weights 'α' should be encoded as (B,N) or (B,N,1) tensors."
                    )
                if β.shape[2] > 1:
                    raise ValueError(
                        "With batches, input weights 'β' should be encoded as (B,M) or (B,M,1) tensors."
                    )
                α, β = α.squeeze(-1), β.squeeze(-1)

            if l_x is not None:
                raise NotImplementedError(
                    'The "multiscale" backend has not been implemented with batches.'
                )
            if l_y is not None:
                raise NotImplementedError(
                    'The "multiscale" backend has not been implemented with batches.'
                )

            B2, N2 = α.shape
            B3, M2 = β.shape
            if B != B2:
                raise ValueError(
                    "Samples 'x' and weights 'α' should have the same batchsize."
                )
            if B != B3:
                raise ValueError(
                    "Samples 'y' and weights 'β' should have the same batchsize."
                )

        else:
            raise ValueError(
                "Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors."
            )

        if N != N2:
            raise ValueError(
                "Weights 'α' and samples 'x' should have compatible shapes."
            )
        if M != M2:
            raise ValueError(
                "Weights 'β' and samples 'y' should have compatible shapes."
            )

        return B, N, M, D, l_x, α, l_y, β