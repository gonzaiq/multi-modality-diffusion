
from diffusers import DDPMScheduler, UNet2DConditionModel
import torch
from typing import List, Optional

from src.correlated_noise import generate_channel_correlated_noise

class CustomDDPMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_noise(self, original_sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Custom noising function."""
        return super().add_noise(original_sample, noise, timesteps)  # Modify here if needed
    
    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """Custom denoising step."""
        return super().step(model_output, timestep, sample, **kwargs)  # Modify here if needed
    
    def sample(self, model, num_inference_steps: int = 50, batch_size: int = 1, shape: Optional[tuple] = None, noise_corr: float = 0.0):
        """Custom sampling function."""
        device = model.device
        #sample = torch.randn((batch_size, *shape), device=device)
        sample = generate_channel_correlated_noise(sample.shape, correlation=noise_corr).to(model.device)
        
        self.set_timesteps(num_inference_steps)
        for t in self.timesteps:
            with torch.no_grad():
                model_output = model(sample, t).sample
            sample = self.step(model_output, t, sample).prev_sample
        
        return sample

class ConditionalDDPMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_noise(self, original_sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Custom noising function."""
        return super().add_noise(original_sample, noise, timesteps) 
    
    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """Custom denoising step."""
        return super().step(model_output, timestep, sample, **kwargs)
    
    def sample(self, model, num_inference_steps: int = 50, batch_size: int = 1, shape: Optional[tuple] = None, noise_corr: float = 0.0):
        """Custom sampling function."""

        # sample correlated noise
        x_t = generate_channel_correlated_noise((batch_size, *shape), correlation=noise_corr).to(model.device).half()
        
        # simulate no conditioning with zero embedding
        seq_len = 320  # Depends on the model
        hidden_dim = 768  # Depends on the model  
        encoder_hidden_states = torch.zeros((batch_size, seq_len, hidden_dim), device=x_t.device).half()

        # Set model to eval mode
        model.eval()

        self.set_timesteps(num_inference_steps)
        with torch.no_grad():
            for t in self.timesteps:
                print(f"time {t}, {x_t.shape}")
                noise_pred = model(x_t, t, encoder_hidden_states=encoder_hidden_states).sample
                x_t = self.step(noise_pred, t, x_t).prev_sample
        
        return x_t.clamp(-1, 1)  # Ensure valid pixel range
