import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import torch
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.correlated_noise import generate_channel_correlated_noise

figures_pth = "figures"

def autocorrelation_2d(x):
    """
    Computes the 2D autocorrelation of a tensor over spatial dimensions.
    Args:
        x (torch.Tensor): The input tensor (H x W).
    Returns:
        torch.Tensor: The 2D autocorrelation.
    """
    # Subtract the mean
    x = x - torch.mean(x)
    # Compute the 2D Fourier Transform
    ft = torch.fft.fft2(x)
    # Compute the 2D autocorrelation using the inverse Fourier Transform
    acf = torch.fft.ifft2(ft * torch.conj(ft)).real
    # Normalize the autocorrelation
    acf /= torch.max(acf)
    return acf

def autocorrelation_channelwise(correlated_noise):
    """
    Computes the autocorrelation across channels and spatial dimensions for the correlated noise.
    Args:
        correlated_noise (torch.Tensor): The input noise tensor with shape (B, C, H, W).
    Returns:
        torch.Tensor: The autocorrelation for each channel and cross-channel correlation.
    """
    B, C, H, W = correlated_noise.shape
    
    # Compute autocorrelation for each channel across the spatial dimensions
    channelwise_acf = []
    for c in range(C):
        acf = autocorrelation_2d(correlated_noise[0, c])  # Compute for the first batch element
        channelwise_acf.append(acf)
    
    # Compute cross-channel correlation across the spatial dimensions
    channel_1_flat = correlated_noise[0, 0].view(-1)
    channel_2_flat = correlated_noise[0, 1].view(-1) 
    channel_3_flat = correlated_noise[0, 2].view(-1) 
    cross_channel_acf = torch.corrcoef(torch.stack([channel_1_flat, channel_2_flat, channel_3_flat]))
    
    return channelwise_acf, cross_channel_acf

def main():
    # Generate channel-correlated noise
    shape = (1, 3, 256, 256)  # (Batch, Channels, Height, Width)
    correlations = [0.0, 0.25, 0.5, 0.75, 0.99]
    
    for correlation in correlations:
        correlated_noise = generate_channel_correlated_noise(shape, correlation)
        
        # Compute autocorrelations
        channelwise_acf, cross_channel_acf = autocorrelation_channelwise(correlated_noise)

        # Plot the generated noise and autocorrelation for each channel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            noise = correlated_noise[0, i].cpu().detach().numpy()
            normalized_image = (noise - noise.min()) / (noise.max() - noise.min())
            axes[i].imshow(normalized_image, cmap='gray')
            axes[i].set_title(f'Channel {i+1}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(figures_pth, f"noise_plot_corr_{correlation}.png"))
        
        # Plot autocorrelation for each channel
        fig_acf, axes_acf = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            axes_acf[i].imshow(channelwise_acf[i].cpu().detach().numpy(), cmap='hot')
            axes_acf[i].set_title(f'ACF Channel {i+1}')
            axes_acf[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(figures_pth, f"acf_plot_corr_{correlation}.png"))
        
        # Plot cross-channel autocorrelation
        fig_cross_acf, ax_cross_acf = plt.subplots(figsize=(6, 5))
        #cax = ax_cross_acf.matshow(cross_channel_acf.cpu().detach().numpy(), cmap='coolwarm')
        cax = ax_cross_acf.matshow(
            cross_channel_acf.cpu().detach().numpy(), 
            cmap='coolwarm', 
            norm=mcolors.Normalize(vmin=0, vmax=1)  # Normalize between 0 and 1
        )
        fig_cross_acf.colorbar(cax)
        ax_cross_acf.set_title(f'Cross-channel Correlation (Corr={correlation})')
        plt.savefig(os.path.join(figures_pth, f"cross_acf_plot_corr_{correlation}.png"))

if __name__ == "__main__":
    main()