from typing import *
from torch import nn, Tensor
import torch


def apply(x: Tensor, T: Tensor) -> Tensor:
    """
    Apply an affine transformation to a set of points.

    Args:
        x (Tensor): The set of points of shape (batch_size, n_points, n_dims).
        T (Tensor): The augmented transformation matrix of shape (batch_size, n_dims + 1, n_dims + 1).
    """

    # Unsqueeze batch dimension if it doesn't exist
    if x.ndim == 2:
        x = x.unsqueeze(0)

    if T.ndim == 2:
        T = T.unsqueeze(0)
        
    x = torch.cat((x, torch.ones(*x.shape[:-1], 1)), dim=-1)

    return (x @ T)[:, :, :-1]

def compute_affine_transform(
                source: Tensor, 
                target: Tensor
                ) -> Tensor:

        """
        Find the optimal affine transformation matrix that maps
        `source` to `target` in the least-squares sense, using Kabsch algorithm.

        See https://en.wikipedia.org/wiki/Kabsch_algorithm for more details and notation.
        
        Args:
            source (Tensor): The source points of shape (batch_size, n_points, n_dims).
            target (Tensor): The target points of shape (n_points, n_dims).
            
        Returns:
            Tensor: The augmented transformation matrix of shape (batch_size, n_dims + 1, n_dims + 1).
            
        """

        batch_size, n_points, n_dims = source.shape

        # Repeat target for each source
        P = target.clone().repeat(batch_size, 1, 1)
        Q = source.clone()

        # Compute means
        P_mean = P.mean(dim=1).unsqueeze(-2) # (batch_size, 1, n_dims)
        Q_mean = Q.mean(dim=1).unsqueeze(-2) # (batch_size, 1, n_dims)
        
        # Compute covariance matrix
        P -= P_mean # (batch_size, n_points, n_dims)
        Q -= Q_mean # (batch_size, n_points, n_dims)

        # Compute scale
        P_scale = P.square().sum(dim=-1).sqrt().sum(dim=-1).view(-1, 1, 1) # (batch_size, 1, 1)
        Q_scale = Q.square().sum(dim=-1).sqrt().sum(dim=-1).view(-1, 1, 1) # (batch_size, 1, 1)

        scale = Q_scale / P_scale

        # Scale
        P /= P_scale # (batch_size, n_points, n_dims)
        Q /= Q_scale # (batch_size, n_points, n_dims)

        H = P.transpose(1, 2) @ Q # (batch_size, n_dims, n_dims)

        # Singular value decomposition
        U, S, V = torch.linalg.svd(H)
        V = V.transpose(1, 2)

        # Ensure a right-handed coordinate system
        d = torch.sign(torch.det(V @ U.transpose(1, 2)))

        # Compute the optimal rotation matrix
        I = torch.eye(n_dims, device=P.device).unsqueeze(0).repeat(P.shape[0], 1, 1)
        I[:, -1, -1] = d

        R = (V @ I @ U.transpose(1, 2)) / scale # (batch_size, n_dims, n_dims)

        # A = torch.cat([R, target_mean.transpose(1,2) -  R @ source_mean.transpose(1,2)], dim=-1)

        # transform = lambda x: x @ A[:, :, :-1] + A[:, :, -1].unsqueeze(1)

        T = torch.eye(n_dims + 1, device=P.device).unsqueeze(0).repeat(P.shape[0], 1, 1)
        T[:, :-1, :-1] = R
        T[:, :-1, -1] = Q_mean.squeeze() - (R @ P_mean.squeeze().unsqueeze(-1)).squeeze()

        # transform = lambda x: (torch.cat((x, torch.ones(*x.shape[:-1], 1)), dim=-1) @ T)[:, :, :-1]
        
        return T