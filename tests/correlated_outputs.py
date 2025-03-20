import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import torch
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from PIL import Image
import torch
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.correlated_noise import generate_channel_correlated_noise
from tests.correlated_images import autocorrelation_channelwise

figures_dir = "figures/ddpm"

def main():
    # Generate channel-correlated noise
    shape = (1, 3, 256, 256)  # (Batch, Channels, Height, Width)
    correlations = np.arange(0.0, 1.0, 0.1)

    for corr in correlations:

        # Load the image
        image_fname = f"output_ddpm_corr_{corr}.png"
        image = Image.open( os.path.join(figures_dir, image_fname))

        # Define the transformation to convert the image to a tensor
        transform = transforms.ToTensor()

        # Apply the transformation
        image_tensor = transform(image).unsqueeze(0)
        # Compute autocorrelations
        channelwise_acf, cross_channel_acf = autocorrelation_channelwise(image_tensor)

        # Plot the generated noise and autocorrelation for each channel
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            noise = image_tensor[0, i].cpu().detach().numpy()
            normalized_image = (noise - noise.min()) / (noise.max() - noise.min())
            axes[i].imshow(normalized_image, cmap='gray')
            axes[i].set_title(f'Channel {i+1}')
            axes[i].axis('off')

        # Create the new image path with the "corr" suffix
        base_name, ext = os.path.splitext(image_fname)
        new_image_fname = f"{base_name}_correlations{ext}"

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, new_image_fname))
        
        # Plot autocorrelation for each channel
        fig_acf, axes_acf = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            axes_acf[i].imshow(channelwise_acf[i].cpu().detach().numpy(), cmap='hot')
            axes_acf[i].set_title(f'ACF Channel {i+1}')
            axes_acf[i].axis('off')

        plt.tight_layout()
        # Create the new image path with the "corr" suffix
        base_name, ext = os.path.splitext(image_fname)
        new_image_fname = f"{base_name}_acf_plot_corr{ext}"
        plt.savefig(os.path.join(figures_dir, new_image_fname))
        
        # Plot cross-channel autocorrelation
        fig_cross_acf, ax_cross_acf = plt.subplots(figsize=(6, 5))
        #cax = ax_cross_acf.matshow(cross_channel_acf.cpu().detach().numpy(), cmap='coolwarm')
        cax = ax_cross_acf.matshow(
            cross_channel_acf.cpu().detach().numpy(), 
            cmap='coolwarm', 
            norm=mcolors.Normalize(vmin=0.6, vmax=1)  # Normalize between 0 and 1
            #norm=mcolors.LogNorm(vmin=0.1, vmax=1)  # Adjust vmin and vmax based on your data range
        )
        fig_cross_acf.colorbar(cax)
        ax_cross_acf.set_title(f'Cross-channel Correlation (Corr={corr})')
        base_name, ext = os.path.splitext(image_fname)
        new_image_fname = f"{base_name}_cross_acf_plot_corr{ext}"
        plt.savefig(os.path.join(figures_dir, new_image_fname))

if __name__ == "__main__":
    main()