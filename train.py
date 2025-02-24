import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from datasets import load_dataset
from torchvision import transforms
from typing import Optional
from src.ddpm_scheduler import ConditionalDDPMScheduler
from src.correlated_noise import generate_channel_correlated_noise
from src.datasets import CityscapesDataset
from src.pipeline import CustomStableDiffusionPipeline
import os

# Training Loop
def train(model, scheduler, dataset, num_epochs=10, batch_size=4, noise_corr: float = 0.8):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:

            # concatenate image and mask to construct input
            images = batch["image"].to(model.device)
            masks = batch["mask"].to(model.device)
            input = torch.cat((images, masks), dim=1)

            # add corelated noise
            noise = generate_channel_correlated_noise(input.shape, correlation=noise_corr).to(model.device)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (input.shape[0],), device=model.device)
            
            noisy_input = scheduler.add_noise(input, noise, timesteps)
            noisy_input = noisy_input.to(dtype=torch.float16)
            model_output = model(noisy_input, timesteps, encoder_hidden_states=None).sample
            loss = F.mse_loss(model_output, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

def main():

    # data directories
    data_pth = "data/Cityscapes"
    img_dir = os.path.join(data_pth, "leftImg8bit_trainvaltest", "leftImg8bit")
    mask_dir = os.path.join(data_pth, "gtFine_trainvaltest", "gtFine")

    # define transforms. Not normalize mask
    transform_image = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to a fixed size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to a fixed size
        transforms.ToTensor(),  # Convert to tensor
    ])

    dataset = CityscapesDataset(img_dir, mask_dir, "train", transform_image, transform_mask)


    # Load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)

    # Extract UNet model
    model = pipe.unet
    model.config.in_channels = 6  # Example: RGB (3) + Segmentation (3)
    model.register_to_config(in_channels=6)
    model.conv_in = torch.nn.Conv2d(in_channels=6, out_channels=320, kernel_size=3, stride=1, padding=1)
    model = model.half()

    # Define DDPM scheduler
    scheduler = ConditionalDDPMScheduler(num_train_timesteps=1000)

    # Run Training
    print(f"Training...")
    train(model, scheduler, dataset, noise_corr=0.8)

if __name__ == "__main__":
    main()