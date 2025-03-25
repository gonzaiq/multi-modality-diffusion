import torch
from torchvision import transforms
from PIL import Image
import os
from diffusers import StableDiffusionPipeline
from src.ddpm_scheduler import ConditionalDDPMScheduler, CustomDDPMScheduler
from diffusers import DDPMScheduler, UNet2DConditionModel
from src.pipelines.stable_diffusion_pipeline import CustomStableDiffusionPipeline
from src.pipelines.ddpm_pipeline import CustomDDPMPipeline
import numpy as np
import matplotlib.pyplot as plt


def main():

    pipeline = "ddpm"

    # Load SD pipeline and model
    if pipeline == "sd":
        pipe = CustomStableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        pipe_args = {"prompt": prompt}
    elif pipeline == "ddpm":
        pipe = CustomDDPMPipeline.from_pretrained("google/ddpm-cat-256")
        pipe.scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256", prediction_type='epsilon')
        pipe_args = {}
    else:
        raise NotImplementedError(f"Pipeline {pipeline} is not implemented")
    
    # Define DDPM scheduler
    scheduler = CustomDDPMScheduler(num_train_timesteps=1000)
    #pipe.scheduler = scheduler

    # Set a fixed seed for reproducibility
    seed = 42  # Choose any integer
    generator = torch.Generator("cpu").manual_seed(seed)

    # Generate an image from a text prompt
    prompt = "A futuristic cityscape at sunset, highly detailed, cinematic lighting"
    correlations = np.arange(0.0, 1.0, 0.1)
    for corr in correlations:

        pipe_args.update(
            {
            "generator": generator,
            "num_inference_steps": 50,
            "noise_corr": corr
            })

        image = pipe(**pipe_args).images[0]

        # Save or display the image
        image.save(f"output_{pipeline}_corr_{corr}.png")
        plt.close()

    # SAMPLE
    """
    image = scheduler.sample(model, shape=(4, 64, 64), num_inference_steps=1000)

    # Convert tensor to image
    image = image[0].permute(1, 2, 0)
    image = image.cpu().numpy()
    image = (image * 255).astype(np.uint8)  # Scale and convert to uint8
    image = Image.fromarray(image)  # Convert to image format
    image.show()

    # Optionally, save the image
    image.save("generated_image.png")
    """
if __name__ == "__main__":
    main()
