import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from datasets import load_dataset
from torchvision import transforms
from typing import Optional
from src.ddpm_scheduler import CustomDDPMScheduler

# Training Loop
def train(model, scheduler, dataset, num_epochs=10, batch_size=4):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            images = batch["image"].to(model.device)
            noise = torch.randn_like(images).to(model.device)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (images.shape[0],), device=model.device)
            
            noisy_images = scheduler.add_noise(images, noise, timesteps)
            model_output = model(noisy_images, timesteps).sample
            loss = F.mse_loss(model_output, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

def preprocess(sample):
    rgb = transform(sample["image"])
    segmentation = transform(sample["segmentation"])
    multimodal_input = torch.cat([rgb, segmentation], dim=0)
    return {"image": multimodal_input}

def main():

    # Load Multimodal Dataset
    dataset = load_dataset("cityscapes", split="train")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = dataset.map(preprocess)

    # Define model and scheduler
    model = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-unet")
    model.config.in_channels = 6  # Example: RGB (3) + Segmentation (3)
    model.register_to_config(in_channels=6)

    scheduler = CustomDDPMScheduler(num_train_timesteps=1000)

    # Run Training
    train(model, scheduler, dataset)