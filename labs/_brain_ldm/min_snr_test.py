import torch

import diffusers
from diffusers import DDPMScheduler
from diffusers.training_utils import compute_snr

noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

timesteps = torch.tensor([0,249,499,749,999]) #torch.randint(0, noise_scheduler.config.num_train_timesteps, (4,))

timesteps = timesteps.long()

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


snr = compute_snr(noise_scheduler, timesteps)



snr_gamma = 5
base_weight_min = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
snr_gamma = 1
base_weight_max = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).max(dim=1)[0]
mse_loss_weights = base_weight / snr

# For zero-terminal SNR, we have to handle the case where a sigma of Zero results in a Inf value.
# When we run this, the MSE loss weights for this timestep is set unconditionally to 1.
# If we do not run this, the loss value will go to NaN almost immediately, usually within one step.
mse_loss_weights[snr == 0] = 1.0

# We first calculate the original loss. Then we mean over the non-batch dimensions and
# rebalance the sample-wise losses with their respective loss weights.
# Finally, we take the mean of the rebalanced loss.
# loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
# loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
# loss = loss.mean()

noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
timesteps
snr
base_weight_min # snr_gamma=5
base_weight_max # snr_gamma=1
base_weight_min / snr
base_weight_max / snr
