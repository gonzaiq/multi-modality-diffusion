
from diffusers import DDPMScheduler, UNet2DConditionModel
import torch
from typing import List, Optional

class CustomDDPMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_noise(self, original_sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Custom noising function."""
        return super().add_noise(original_sample, noise, timesteps)  # Modify here if needed
    
    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """Custom denoising step."""
        return super().step(model_output, timestep, sample, **kwargs)  # Modify here if needed
    
    def sample(self, model, num_inference_steps: int = 50, batch_size: int = 1, shape: Optional[tuple] = None):
        """Custom sampling function."""
        device = model.device
        sample = torch.randn((batch_size, *shape), device=device)
        
        self.set_timesteps(num_inference_steps)
        for t in self.timesteps:
            with torch.no_grad():
                model_output = model(sample, t).sample
            sample = self.step(model_output, t, sample).prev_sample
        
        return sample

class ConditionalCustomDDPMScheduler(DDPMScheduler):
    def __init__(self, *args, n_modalities: int = 1, modality_groups: Optional[List[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if n_modalities > 1:
            assert modality_groups is not None, "If more than one modality is selected, modalitiy groups must be defined"
            assert n_modalities == len(modality_groups), "Lengh of modality groups should match the number of modalities"
        
        self.n_modalities = n_modalities
        self.modality_groups = modality_groups
    
    def add_noise(self, original_sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Custom noising function."""
        return super().add_noise(original_sample, noise, timesteps)  # Modify here if needed
    
    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """Custom denoising step."""
        return super().step(model_output, timestep, sample, **kwargs)  # Modify here if needed
    
    def sample(self, model, num_inference_steps: int = 50, batch_size: int = 1, shape: Optional[tuple] = None):
        """Custom sampling function."""
        device = model.device
        sample = torch.randn((batch_size, *shape), device=device)
        
        self.set_timesteps(num_inference_steps)
        for t in self.timesteps:
            with torch.no_grad():
                model_output = model(sample, t).sample
            sample = self.step(model_output, t, sample).prev_sample
        
        return sample
