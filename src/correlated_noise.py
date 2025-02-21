import torch

def generate_channel_correlated_noise(shape, correlation=0.5):
    """
    Generates noise with correlation across the channel dimension.
    
    Args:
        shape (tuple): (Batch, Channels, Height, Width)
        correlation (float): Degree of correlation between channels (0 = independent, 1 = identical).
    
    Returns:
        torch.Tensor: Correlated noise tensor.
    """
    B, C, H, W = shape  # Batch, Channels, Height, Width
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define a covariance matrix for the channels
    cov_matrix = torch.full((C, C), correlation, device=device)  # Fill with correlation value
    cov_matrix.fill_diagonal_(1.0)  # Set diagonal to 1 (variance of each channel)

    # Compute Cholesky decomposition (L such that L @ L.T = cov_matrix)
    L = torch.linalg.cholesky(cov_matrix)  # Lower triangular matrix

    # Sample uncorrelated Gaussian noise for all channels
    noise = torch.randn(B, C, H, W, device=device)  # Independent noise

    # Apply Cholesky transformation to introduce correlation
    correlated_noise = torch.einsum("ij,bcij->bcij", L, noise)  # Matrix multiplication per batch
    
    return correlated_noise

# Usage in training loop
noise = generate_channel_correlated_noise(images.shape, correlation=0.7).to(model.device)