import matplotlib.pyplot as plt

def main():
    # Generate channel-correlated noise
    shape = (1, 3, 256, 256)  # (Batch, Channels, Height, Width)
    correlation = 0.7
    correlated_noise = generate_channel_correlated_noise(shape, correlation)

    # Plot the generated noise for each channel
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axes[i].imshow(correlated_noise[0, i].cpu().detach().numpy(), cmap='gray')
        axes[i].set_title(f'Channel {i+1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()