import torch

def generate_channel_correlated_noise(shape, correlation=0.0, generator=None, device=None, dtype=None, layout=torch.strided,):
    """
    Generates noise with correlation across the channel dimension.
    
    Args:
        shape (tuple): (Batch, Channels, Height, Width)
        correlation (float): Degree of correlation between channels (0 = independent, 1 = identical).
    
    Returns:
        torch.Tensor: Correlated noise tensor.
    """
    print("-----------------------")
    print("-----------------------")
    print("-----------------------")
    print("-----------------------")
    print("-----------------------")
    print("-----------------------")
    print("-----------------------")
    print("-----------------------")
    print("-----------------------v")
    print(f"go correlation: {correlation}, generator={generator}")
    B, C, H, W = shape  # Batch, Channels, Height, Width
    print(B, C, H, W)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define a covariance matrix for the channels
    cov_matrix = torch.full((C, C), correlation, device=device)  # Fill with correlation value
    cov_matrix.fill_diagonal_(1.0)  # Set diagonal to 1 (variance of each channel)

    # Compute Cholesky decomposition (L such that L @ L.T = cov_matrix)
    L = torch.linalg.cholesky(cov_matrix) # Lower triangular matrix

    # Sample uncorrelated Gaussian noise for all channels
    print("-------")
    print("-------")
    print("-------")
    print(B, C, H, W)
    noise = torch.randn(B, C, H, W, generator=generator, device=device, dtype=dtype, layout=layout)  # Independent noise
    noise = noise.view(B, C, -1)  # Reshape to (B, C, H*W)

    correlated_noise = torch.matmul(L, noise) # Shape: (B, C, H*W)
    correlated_noise = correlated_noise.view(B, C, H, W)  # Reshape back to (B, C, H, W)
    
    return correlated_noise