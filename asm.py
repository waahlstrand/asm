import torch
from torch import Tensor, nn
from typing import *
from .utils import apply, compute_affine_transform

class ActiveShapeModel(nn.Module):

    def __init__(self, n_modes: int) -> None:
        super().__init__()

        self.n_modes = n_modes

        self.prototype: Tensor = None

    @torch.no_grad()
    def forward(self, points: Tensor) -> Tensor:

        batch_size, n_points, n_dims = points.shape

        # Standardize
        points = self.standardize(points.clone())

        # Align
        aligned, T = self.align(self.prototype, points)

        # Compute the deviation
        deviation = (aligned - self.prototype).reshape(n_points * n_dims, -1)

        # Solve for the model parameters
        U, S, V = torch.linalg.svd(aligned.reshape(-1, n_points * n_dims))

        eigenvectors = V.T[:, :self.n_modes]  
        params = torch.linalg.lstsq(eigenvectors, deviation).solution

        # Reconstruct the shape
        new_shape = self.prototype + (eigenvectors @ params).reshape(-1, n_points, n_dims)

        return aligned, new_shape

    @torch.no_grad()
    def fit(self, points: Tensor, tol: float = 1e-6) -> None:

        points  = self.standardize(points.clone())
        mu_old  = self.prototype_from_points(points)
        aligned = points.clone()

        while True:

            # Align
            aligned, T = self.align(mu_old, aligned)

            # Update prototype
            mu_new = self.prototype_from_points(aligned)

            # Check for convergence
            mu_new = self.standardize(mu_new)

            # Change
            delta = (mu_new - mu_old).square().sum(dim=-1).sqrt().sum(dim=-1).max()

            if delta < tol:
                break

            mu_old = mu_new

        self.prototype = mu_new

        return mu_new, aligned

    def prototype_from_points(self, points: Tensor) -> Tensor:

        """
        Compute the mean shape from a set of shapes.
        
        Args:
            points (Tensor): The set of shapes of shape (batch_size, n_points, n_dims).
            
        Returns:
            Tensor: The mean shape of shape (batch_size, n_points, n_dims).
            
        """
        
        prototype = points.mean(dim=0)
        prototype /= prototype.square().sum(dim=-1).sqrt().sum(dim=-1)

        prototype = prototype.unsqueeze(0)

        return prototype
    
    def align(self, prototype: Tensor, points: Tensor) -> Tensor:

        T = compute_affine_transform(points, prototype)

        return apply(points, T), T
    
    def unalign(self, points: Tensor, T: Tensor) -> Tensor:

        T = T.transpose(1, 2)

        return apply(points, T)

    def standardize(self, points: Tensor) -> Tensor:
        """
        Standardize a set of shapes.
        
        Args:
            points (Tensor): The set of shapes of shape (batch_size, n_points, n_dims).
            
        Returns:
            Tensor: The standardized set of shapes of shape (batch_size, n_points, n_dims).
            
        """
        
        batch_size, n_points, n_dims = points.shape
        
        # Compute means
        points_mean = points.mean(dim=1).unsqueeze(-2) # (batch_size, 1, n_dims)
        
        # Compute covariance matrix
        points -= points_mean # (batch_size, n_points, n_dims)
        
        # Compute scale
        points_scale = points.square().sum(dim=-1).sqrt().sum(dim=-1).view(-1, 1, 1) # (batch_size, 1, 1)
        
        points /= points_scale # (batch_size, n_points, n_dims)
        
        return points



def optimize(points: Tensor, prototype: Tensor, tol: float, n_iter: int) -> Tuple[Tensor, Tensor]:

    mu_old  = compute_prototype(points)
    # mu_old  = standardize(mu_old)
    aligned = points.clone()

    while True:

        # Align
        aligned, T = align(mu_old, aligned)

        # Update prototype
        mu_new = compute_prototype(aligned)

        # Check for convergence
        mu_new = standardize(mu_new)

        # Change
        delta = (mu_new - mu_old).square().sum(dim=-1).sqrt().sum(dim=-1).max()

        if delta < tol:
            break

        mu_old = mu_new

    return mu_new, aligned, T


def align(prototype: Tensor, points: Tensor) -> Tuple[Tensor, Tensor]:

    T = compute_affine_transform(points, prototype)

    return apply(points, T), T


def standardize(points: Tensor) -> Tensor:
    """
    Standardize a set of shapes.
    
    Args:
        points (Tensor): The set of shapes of shape (batch_size, n_points, n_dims).
        
    Returns:
        Tensor: The standardized set of shapes of shape (batch_size, n_points, n_dims).
        
    """
    
    batch_size, n_points, n_dims = points.shape
    
    # Compute means
    points_mean = points.mean(dim=1).unsqueeze(-2) # (batch_size, 1, n_dims)
    
    # Compute covariance matrix
    points -= points_mean # (batch_size, n_points, n_dims)
    
    # Compute scale
    points_scale = points.square().sum(dim=-1).sqrt().sum(dim=-1).view(-1, 1, 1) # (batch_size, 1, 1)
    
    points /= points_scale # (batch_size, n_points, n_dims)
    
    return points

def compute_prototype(points: Tensor) -> Tensor:
    """
    Compute the mean shape from a set of shapes.
    
    Args:
        points (Tensor): The set of shapes of shape (batch_size, n_points, n_dims).
        
    Returns:
        Tensor: The mean shape of shape (batch_size, n_points, n_dims).
        
    """
    
    prototype = points.mean(dim=0)
    prototype /= prototype.square().sum(dim=-1).sqrt().sum(dim=-1)

    prototype = prototype.unsqueeze(0)

    return prototype

