import torch
from torchvision import transforms
from PIL import Image
import os
from diffusers import StableDiffusionPipeline
from src.ddpm_scheduler import ConditionalDDPMScheduler, CustomDDPMScheduler
from diffusers import DDPMScheduler, UNet2DConditionModel
from src.pipeline import CustomStableDiffusionPipeline
import numpy as np

# Main function
def main():
    # Data directories (Replace these with actual paths)
    data_pth = "data/Cityscapes"
    img_dir = os.path.join(data_pth, "leftImg8bit_trainvaltest", "leftImg8bit")
    mask_dir = os.path.join(data_pth, "gtFine_trainvaltest", "gtFine")

    # Define transforms
    transform_image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load SD pipeline and model
    pipe = CustomStableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    #pipe = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch")
    print(pipe)

    model = pipe.unet

    # Modify the U-Net model
    #model.config.in_channels = 6  # RGB + Segmentation
    #model.register_to_config(in_channels=6)
    #model.conv_in = torch.nn.Conv2d(in_channels=6, out_channels=320, kernel_size=3, stride=1, padding=1)
    #model = model.half()

    # Define DDPM scheduler
    scheduler = CustomDDPMScheduler(num_train_timesteps=1000)
    pipe.scheduler = scheduler
    # Generate an image from a text prompt
    prompt = "A futuristic cityscape at sunset, highly detailed, cinematic lighting"
    image = pipe(prompt, noise_corr=0.0).images[0]

    # Save or display the image
    image.save("output.png")
    image.show()

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
