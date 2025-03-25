import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from diffusers import DDPMPipeline, UNet2DConditionModel
from diffusers import DDPMScheduler
import pytorch_lightning as pl
import numpy as np
from src.pipelines.random_tensor import randn_tensor

class CorrelatedNoiseDDPMTrainer(pl.LightningModule):
    def __init__(self, model_name="google/ddpm-cat-256", lr=1e-4, noise_corr: float = 0.1):
        super().__init__()
        self.model = UNet2DConditionModel.from_pretrained(model_name)
        self.scheduler = DDPMScheduler.from_pretrained(model_name)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr
        self.noise_corr = noise_corr

    def forward(self, noisy_images, timesteps):
        return self.model(noisy_images, timesteps).sample

    def training_step(self, batch, batch_idx):

        # sample correlated noise
        noise = randn_tensor(batch.shape, noise_corr=self.noise_corr, device=batch.device, dtype=batch.dtype)

        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (batch.shape[0],), device=self.device)
        noisy_images = self.scheduler.add_noise(batch, noise, timesteps)
        noise_pred = self(noisy_images, timesteps)
        loss = self.loss_fn(noise_pred, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)