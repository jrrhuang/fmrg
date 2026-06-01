"""
FLUX FlowMap sampler for reward-guided generation.

Uses the reward ensemble: HPSv2 + ImageReward + PickScore + CLIP.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import gc
import math
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple
from collections import OrderedDict
import sys
from PIL import Image

# =============================================================================
# CONFIGURABLE PATHS - Modify these for your setup
# =============================================================================

# Directory containing this script (reward/style/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Parent reward/ directory (where symlinks live)
REWARD_ROOT = os.path.dirname(SCRIPT_DIR)

# HuggingFace cache directory (for downloading FLUX.1-dev model)
# Set to your preferred cache location, or use environment variable
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Default FLUX model ID (from HuggingFace)
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"

# Default path to FlowMap LoRA weights
DEFAULT_LORA_PATH = os.environ.get("FMRG_LORA_PATH", os.path.join(SCRIPT_DIR, "checkpoints", "flux-flowmap-lora"))

sys.path.insert(0, SCRIPT_DIR)
from flux_two_timestep import FluxPipelineTwoTimestep, add_dual_time_embedder
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor


# =============================================================================
# REWARD MODEL CLASSES
# =============================================================================


def create_clip_preprocess(image_size: int = 224) -> T.Compose:
    """CLIP-compatible image transform: Resize(bicubic) -> CenterCrop -> Normalize."""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])


# =============================================================================
# reward ensemble: HPSv2 + ImageReward + PickScore + CLIP.
# =============================================================================

# Default cache directory for reward model downloads
REWARD_CACHE_DIR = os.environ.get("HF_HOME", os.path.join("/scratch", os.environ.get("USER", "user"), ".cache/huggingface"))


class HPSReward(nn.Module):
    """HPSv2 reward model. Loss = 1 - cosine_similarity(image, text)."""

    def __init__(self, device="cuda", dtype=torch.float32, cache_dir=REWARD_CACHE_DIR):
        super().__init__()
        from hpsv2.src.open_clip import create_model, get_tokenizer
        import huggingface_hub

        self.model = create_model("ViT-H-14", "laion2B-s32B-b79K", precision=dtype, device=device, cache_dir=cache_dir)
        ckpt_path = huggingface_hub.hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt", cache_dir=cache_dir)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
        self.tokenizer = get_tokenizer("ViT-H-14")
        self.model = self.model.to(device, dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.set_grad_checkpointing(True)
        self.device = device

    def forward(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """image: preprocessed [B,3,224,224]. Returns scalar loss (lower=better)."""
        img_features = self.model.encode_image(image)
        text = self.tokenizer(prompt).to(self.device)
        txt_features = self.model.encode_text(text)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        similarity = img_features @ txt_features.T
        return 1 - torch.diagonal(similarity)[0]


class ImageRewardReward(nn.Module):
    """ImageReward model. Loss = 2 - score."""

    def __init__(self, device="cuda", dtype=torch.float32, cache_dir=REWARD_CACHE_DIR):
        super().__init__()
        import ImageReward as RM

        self.model = RM.load("ImageReward-v1.0", download_root=cache_dir)
        self.model = self.model.to(device=device, dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.dtype = dtype

    def forward(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """image: preprocessed [B,3,224,224]. Returns scalar loss (lower=better)."""
        text_input = self.model.blip.tokenizer(
            prompt, padding="max_length", truncation=True, max_length=35, return_tensors="pt"
        ).to(self.device)
        image_embeds = self.model.blip.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.model.blip.text_encoder(
            text_input.input_ids, attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].to(self.device, dtype=self.dtype)
        rewards = self.model.mlp(txt_features)
        rewards = (rewards - self.model.mean) / self.model.std
        return (2 - rewards).mean()


class CLIPReward(nn.Module):
    """CLIP ViT-H reward. Loss = 100 - logit_scale * cosine_similarity."""

    def __init__(self, device="cuda", dtype=torch.float32, cache_dir=REWARD_CACHE_DIR, tokenizer=None):
        super().__init__()
        from transformers import CLIPModel

        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir)
        self.model = self.model.to(device, dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.gradient_checkpointing_enable()
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """image: preprocessed [B,3,224,224]. Returns scalar loss (lower=better)."""
        img_features = self.model.get_image_features(image)
        prompt_token = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=77, truncation=True).to(self.device)
        txt_features = self.model.get_text_features(**prompt_token)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        return 100 - (img_features @ txt_features.T).mean() * self.model.logit_scale.exp()


class PickScoreReward(nn.Module):
    """PickScore reward. Loss = 30 - logit_scale * cosine_similarity."""

    def __init__(self, device="cuda", dtype=torch.float32, cache_dir=REWARD_CACHE_DIR, tokenizer=None):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1", cache_dir=cache_dir).eval()
        self.model = self.model.to(device, dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model._set_gradient_checkpointing(True)
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """image: preprocessed [B,3,224,224]. Returns scalar loss (lower=better)."""
        img_features = self.model.get_image_features(image)
        prompt_token = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=77, truncation=True).to(self.device)
        txt_features = self.model.get_text_features(**prompt_token)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        return 30 - (self.model.logit_scale.exp() * (img_features @ txt_features.T)).mean()


class RewardEnsemble:
    """
    Ensemble of 4 reward models.
    Computes weighted sum of losses. All models share the same CLIP preprocessing.

    Usage:
        ensemble = RewardEnsemble(device="cuda", cache_dir="/scratch/user/.cache/huggingface")
        loss, losses_dict = ensemble(image_01, prompt)  # image in [0,1], [B,3,H,W]
    """

    def __init__(self, device="cuda", dtype=torch.float32, cache_dir=REWARD_CACHE_DIR,
                 hps_weight=5.0, imagereward_weight=1.0, pickscore_weight=0.05, clip_weight=0.01):
        from transformers import AutoProcessor

        self.device = device
        self.dtype = dtype
        self.preprocess = create_clip_preprocess(224)
        self.weights = {
            "hps": hps_weight,
            "imagereward": imagereward_weight,
            "pickscore": pickscore_weight,
            "clip": clip_weight,
        }

        # Load shared tokenizer for CLIP and PickScore
        print("Loading reward model tokenizer...")
        tokenizer = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir)

        # Load reward models
        self.models = {}
        if hps_weight > 0:
            print(f"Loading HPSv2 (weight={hps_weight})...")
            self.models["hps"] = HPSReward(device=device, dtype=dtype, cache_dir=cache_dir)
        if imagereward_weight > 0:
            print(f"Loading ImageReward (weight={imagereward_weight})...")
            self.models["imagereward"] = ImageRewardReward(device=device, dtype=dtype, cache_dir=cache_dir)
        if clip_weight > 0:
            print(f"Loading CLIP (weight={clip_weight})...")
            self.models["clip"] = CLIPReward(device=device, dtype=dtype, cache_dir=cache_dir, tokenizer=tokenizer)
        if pickscore_weight > 0:
            print(f"Loading PickScore (weight={pickscore_weight})...")
            self.models["pickscore"] = PickScoreReward(device=device, dtype=dtype, cache_dir=cache_dir, tokenizer=tokenizer)

        print(f"RewardEnsemble loaded: {list(self.models.keys())}")

    def __call__(self, image_01: torch.Tensor, prompt: str) -> tuple:
        """
        Compute weighted ensemble loss.

        Args:
            image_01: Image tensor in [0, 1] range, shape [B, 3, H, W]
            prompt: Text prompt string

        Returns:
            total_loss: Weighted sum of all losses (differentiable)
            losses_dict: Dict of individual loss values {name: float}
        """
        preprocessed = self.preprocess(image_01)
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        losses_dict = {}

        for name, model in self.models.items():
            loss = model(preprocessed, prompt)
            losses_dict[name] = loss.item()
            total_loss = total_loss + self.weights[name] * loss

        losses_dict["total"] = total_loss.item()
        return total_loss, losses_dict


class FluxFlowMapSampler:
    """
    FLUX FlowMap sampler with integrated VAE and model loading
    Uses FluxPipelineTwoTimestep for dual timestep conditioning
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, lora_path: str = DEFAULT_LORA_PATH,
                 device='cuda', dtype=torch.bfloat16, prompt: str = "",
                 additional_prompts: list = None):
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.lora_path = lora_path
        self.prompt_text = prompt  # Store raw prompt text for reward model tokenizers

        torch.backends.cuda.matmul.allow_tf32 = True

        # Load pipeline with prompt precomputation
        self._load_model(prompt=prompt, additional_prompts=additional_prompts)

        # FLUX VAE scaling factor.
        # FLUX VAE: scaling_factor=0.3611, shift_factor=0.1159
        self.vae = self.pipeline.vae
        self.vae_scale_factor = self.pipeline.vae_scale_factor  # 16 for FLUX

    def _load_model(self, prompt: str = "", additional_prompts: list = None):
        """Load the FLUX FlowMap pipeline with LoRA, with memory-efficient text encoding"""
        print(f"Loading FLUX FlowMap pipeline from: {self.model_id}")
        print(f"Loading LoRA weights from: {self.lora_path}")

        self.pipeline = FluxPipelineTwoTimestep.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            cache_dir=HF_CACHE
        ).to(self.device)

        # Add dual time embedder and load LoRA
        self.pipeline.transformer = add_dual_time_embedder(self.pipeline.transformer)
        self.pipeline.load_lora_weights(self.lora_path, weight_name="pytorch_lora_weights.safetensors")

        # Precompute embeddings before deleting text encoders
        print("Precomputing text embeddings for memory efficiency...")
        self.prompt_cache = {}
        with torch.no_grad():
            # Null embeddings (for CFG if needed)
            self.null_prompt_embeds, self.null_pooled_prompt_embeds = self._encode_prompt_internal("")
            # User prompt embeddings
            if prompt:
                self.cached_prompt_embeds, self.cached_pooled_prompt_embeds = self._encode_prompt_internal(prompt)
                self.prompt_cache[prompt] = (self.cached_prompt_embeds.cpu(), self.cached_pooled_prompt_embeds.cpu())
                print(f"Precomputed embeddings for prompt: '{prompt[:50]}...'")
            else:
                # Use null embeddings if no prompt provided
                self.cached_prompt_embeds = self.null_prompt_embeds.clone()
                self.cached_pooled_prompt_embeds = self.null_pooled_prompt_embeds.clone()

            # Precompute additional prompts (for multi-prompt sweeps)
            if additional_prompts:
                for p in additional_prompts:
                    if p not in self.prompt_cache:
                        embeds, pooled = self._encode_prompt_internal(p)
                        self.prompt_cache[p] = (embeds.cpu(), pooled.cpu())
                print(f"Precomputed embeddings for {len(self.prompt_cache)} total prompts")

        # Free text encoders after prompt embeddings are cached.
        print("Removing text encoders to free memory...")
        del self.pipeline.text_encoder
        del self.pipeline.text_encoder_2
        del self.pipeline.tokenizer
        del self.pipeline.tokenizer_2
        self.pipeline.text_encoder = None
        self.pipeline.text_encoder_2 = None
        self.pipeline.tokenizer = None
        self.pipeline.tokenizer_2 = None
        gc.collect()
        torch.cuda.empty_cache()

        print(f"FLUX FlowMap loaded successfully (memory-efficient mode)")

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image [B, 3, H, W] in [-1, 1] to latent [B, C, H/16, W/16]."""
        latents = self.vae.encode(image).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent [B, C, H/16, W/16] back to image [B, 3, H, W] in [-1, 1]. Gradients flow through."""
        z_scaled = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        samples = self.vae.decode(z_scaled, return_dict=False)[0]
        return torch.clamp(samples, -1, 1)

    def decode_no_grad(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decode(z)

    def reward_consistency(self, z0t: torch.Tensor, reward_ensemble: 'RewardEnsemble',
                                 prompt: str, stepsize: float = 0.1, num_iters: int = 3,
                                 lmbda: float = 0.0, timestep_idx: int = 0,
                                 total_timesteps: int = 20, use_adam: bool = True):
        """Euclidean (FMRG-E) update on z0t. Returns (z0t_opt, loss, accumulated_grad)."""
        z0t_original = z0t.clone().detach()
        z0t_opt = z0t.clone().detach().requires_grad_(True)
        loss_val = None
        total_grad = torch.zeros_like(z0t)

        if use_adam:
            lr = stepsize * (1.0 - timestep_idx / total_timesteps)
            optimizer = torch.optim.Adam([z0t_opt], lr=lr)

            for _ in range(num_iters):
                optimizer.zero_grad()
                pred_image = self.decode(z0t_opt).float()
                pred_image_01 = ((pred_image + 1) / 2).clamp(0, 1)

                reward_loss, losses_dict = reward_ensemble(pred_image_01, prompt)
                reg_loss = F.mse_loss(z0t_opt, z0t_original)
                loss = reward_loss + lmbda * reg_loss
                loss_val = reward_loss.item()

                loss.backward()
                optimizer.step()
        else:
            for _ in range(num_iters):
                pred_image = self.decode(z0t_opt).float()
                pred_image_01 = ((pred_image + 1) / 2).clamp(0, 1)

                reward_loss, losses_dict = reward_ensemble(pred_image_01, prompt)
                reg_loss = F.mse_loss(z0t_opt, z0t_original)
                loss = reward_loss + lmbda * reg_loss
                loss_val = reward_loss.item()

                grad = torch.autograd.grad(loss, z0t_opt)[0].to(self.dtype)
                total_grad = total_grad + grad
                z0t_opt = (z0t_opt - stepsize * grad).detach().requires_grad_(True)

        return z0t_opt.detach(), loss_val if loss_val is not None else 0.0, total_grad.detach()

    def reward_consistency_xt(self, zt: torch.Tensor, t_cur: float,
                                    reward_ensemble: 'RewardEnsemble', prompt: str,
                                    prompt_embeds: torch.Tensor, pooled_prompt_embeds: torch.Tensor,
                                    guidance_scale: float = 3.5,
                                    stepsize: float = 0.1, num_iters: int = 3,
                                    lmbda: float = 0.0,
                                    grad_norm_mode: str = "none", velocity_norm: torch.Tensor = None,
                                    velocity: torch.Tensor = None, t_next: float = 0.0,
                                    use_adam: bool = False, timestep_idx: int = 0, total_timesteps: int = 28):
        """
        Reward consistency w.r.t. z_t (Jacobian / FMRG-J branch).
        """
        zt_original = zt.clone().detach()
        loss_val = None
        scaled_grad = None

        if use_adam:
            lr = stepsize * (1.0 - float(timestep_idx) / float(total_timesteps))
            optimizer = torch.optim.Adam([zt], lr=lr)

            for iter_idx in range(num_iters):
                optimizer.zero_grad()
                if iter_idx == 0 and velocity is not None:
                    u = velocity
                else:
                    u = self.predict_vector(zt, t_cur, prompt_embeds=prompt_embeds,
                                            pooled_prompt_embeds=pooled_prompt_embeds,
                                            guidance_scale=guidance_scale, t_next=t_next)
                z0t = zt - t_cur * u
                pred_image = self.decode(z0t).float()
                pred_image_01 = ((pred_image + 1) / 2).clamp(0, 1)

                reward_loss, losses_dict = reward_ensemble(pred_image_01, prompt)
                if lmbda > 0:
                    loss = reward_loss + lmbda * F.mse_loss(zt, zt_original)
                else:
                    loss = reward_loss
                loss_val = reward_loss.item()

                loss.backward()
                optimizer.step()

            grad_zt = -(zt.detach() - zt_original)
            return grad_zt, loss_val if loss_val is not None else 0.0, zt_original

        # Gradient descent fallback
        for iter_idx in range(num_iters):
            if iter_idx == 0 and velocity is not None:
                u = velocity
            else:
                u = self.predict_vector(zt, t_cur, prompt_embeds=prompt_embeds,
                                        pooled_prompt_embeds=pooled_prompt_embeds,
                                        guidance_scale=guidance_scale, t_next=t_next)
            z0t = zt - t_cur * u
            pred_image = self.decode(z0t).float()
            pred_image_01 = ((pred_image + 1) / 2).clamp(0, 1)

            reward_loss, losses_dict = reward_ensemble(pred_image_01, prompt)
            if lmbda > 0:
                loss = reward_loss + lmbda * F.mse_loss(zt, zt_original)
            else:
                loss = reward_loss
            loss_val = reward_loss.item()

            grad = torch.autograd.grad(loss, zt)[0]
            if grad_norm_mode == "normalize":
                grad_norm = torch.norm(grad)
                grad = grad / (grad_norm + 1e-8)
                if velocity_norm is not None:
                    grad = grad * velocity_norm
                zt = (zt - stepsize * grad).detach().requires_grad_(True)
            elif grad_norm_mode == "clip" and velocity_norm is not None:
                norm_cap = 8.0
                scaled_grad = stepsize * grad
                clip_threshold = norm_cap * velocity_norm
                grad_norm = torch.linalg.vector_norm(scaled_grad, dim=(1, 2, 3), keepdim=True)
                factor = torch.clamp(clip_threshold / (grad_norm + 1e-8), max=1.0)
                scaled_grad = scaled_grad * factor
                zt = (zt - scaled_grad).detach().requires_grad_(True)
            else:
                zt = (zt - stepsize * grad).detach().requires_grad_(True)

        if num_iters == 1 and grad_norm_mode == "clip" and scaled_grad is not None:
            grad_zt = scaled_grad.detach()
        else:
            grad_zt = -(zt.detach() - zt_original)
        return grad_zt, loss_val if loss_val is not None else 0.0, zt_original

    def _progress_lookahead_decode(self, z_in: torch.Tensor, t_start_norm: float,
                                    num_steps: int, height: int, width: int,
                                    guidance_scale: float = 3.5) -> torch.Tensor:
        """Run num_steps flow-matching forward from z at t_start (normalized [0,1])
        down to t=0; return decoded image tensor in [-1, 1].

        Uses FLUX's proper mu-shifted schedule (matches sample_forward) so progress
        images are sharp instead of grainy.

        Accepts either packed [B, seq, dim] or unpacked [B, C, H_l, W_l] z.
        Used only for visualization/progress callback — does not affect the main sampler.
        """
        device = z_in.device
        batch_size = z_in.shape[0]
        H_latent = 2 * (int(height) // (self.vae_scale_factor * 2))
        W_latent = 2 * (int(width) // (self.vae_scale_factor * 2))
        z = z_in.detach().to(self.dtype)
        if z.dim() == 4:
            num_channels_latents = z.shape[1]
            z = self.pipeline._pack_latents(z, batch_size, num_channels_latents, H_latent, W_latent)
        latent_image_ids = self.pipeline._prepare_latent_image_ids(
            batch_size, H_latent // 2, W_latent // 2, device, self.dtype)
        text_ids = torch.zeros(self.cached_prompt_embeds.shape[1], 3, device=device, dtype=self.dtype)
        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)
        else:
            guidance = None

        # Build sigmas scaled by t_start_norm so the (mu-shifted) schedule starts at t_start_norm
        # and ends near 0. This mirrors sample_forward's linspace(1, 1/N, N) but scaled.
        ts = max(float(t_start_norm), 1e-4)
        sigmas = np.linspace(ts, ts / int(num_steps), int(num_steps))
        image_seq_len = z.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.get("base_image_seq_len", 256),
            self.pipeline.scheduler.config.get("max_image_seq_len", 4096),
            self.pipeline.scheduler.config.get("base_shift", 0.5),
            self.pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            self.pipeline.scheduler, int(num_steps), device, sigmas=sigmas, mu=mu,
        )
        self.pipeline.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            t_next = timesteps[i + 1] if i < len(timesteps) - 1 else torch.zeros_like(t)
            timestep = t.expand(batch_size).to(z.dtype)
            timestep2 = t_next.expand(batch_size).to(z.dtype)
            v = self.pipeline.transformer(
                hidden_states=z,
                timestep=torch.stack([timestep / 1000, timestep2 / 1000], dim=-1),
                guidance=guidance,
                pooled_projections=self.cached_pooled_prompt_embeds,
                encoder_hidden_states=self.cached_prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            z = self.pipeline.scheduler.step(v, t, z, return_dict=False)[0].to(self.dtype)
        latents_unpacked = self.pipeline._unpack_latents(z, height, width, self.vae_scale_factor)
        latents_scaled = (latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents_scaled, return_dict=False)[0]
        return image

    def predict_velocity_packed(self, z_packed: torch.Tensor, t_cur: torch.Tensor,
                                  t_next: torch.Tensor, guidance_scale: float = 3.5,
                                  latent_image_ids: torch.Tensor = None,
                                  text_ids: torch.Tensor = None) -> torch.Tensor:
        """Velocity from the FLUX FlowMap transformer on already-packed latents."""
        batch_size = z_packed.shape[0]
        device = z_packed.device

        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(batch_size)
        else:
            guidance = None

        # Timesteps: expand to batch and convert to [0, 1] for transformer
        timestep = t_cur.expand(batch_size).to(z_packed.dtype)
        timestep2 = t_next.expand(batch_size).to(z_packed.dtype)

        # Call transformer with both timesteps stacked.
        with self.pipeline.transformer.cache_context("cond"):
            noise_pred = self.pipeline.transformer(
                hidden_states=z_packed,
                timestep=torch.stack([timestep / 1000, timestep2 / 1000], dim=-1),
                guidance=guidance,
                pooled_projections=self.cached_pooled_prompt_embeds,
                encoder_hidden_states=self.cached_prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]

        return noise_pred

    def predict_vector(self, z: torch.Tensor, t_cur: float,
                       prompt_embeds: torch.Tensor = None,
                       pooled_prompt_embeds: torch.Tensor = None,
                       guidance_scale: float = 3.5, t_next: float = None):
        """Velocity from the FLUX FlowMap transformer; t_cur/t_next in [0, 1] (1=noise, 0=data)."""
        batch_size = z.shape[0]
        device = z.device

        if isinstance(t_cur, torch.Tensor):
            t_cur = t_cur.item()
        if t_next is not None and isinstance(t_next, torch.Tensor):
            t_next = t_next.item()
        if t_next is None:
            t_next = t_cur

        # FLUX time convention: t=1 is noise, t=0 is data; transformer expects t·1000.
        timestep = torch.tensor([t_cur * 1000], device=device, dtype=z.dtype).expand(batch_size)
        timestep2 = torch.tensor([t_next * 1000], device=device, dtype=z.dtype).expand(batch_size)

        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)
        else:
            guidance = None

        H_latent, W_latent = z.shape[2], z.shape[3]
        H_image = H_latent * self.vae_scale_factor
        W_image = W_latent * self.vae_scale_factor

        latent_image_ids = self.pipeline._prepare_latent_image_ids(
            batch_size, H_latent // 2, W_latent // 2, device, z.dtype
        )
        z_packed = self.pipeline._pack_latents(z, batch_size, z.shape[1], H_latent, W_latent)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=z.dtype)

        with self.pipeline.transformer.cache_context("cond"):
            noise_pred = self.pipeline.transformer(
                hidden_states=z_packed,
                timestep=torch.stack([timestep / 1000, timestep2 / 1000], dim=-1),
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

        velocity = self.pipeline._unpack_latents(noise_pred, H_image, W_image, self.vae_scale_factor)

        return velocity

    def _encode_prompt_internal(self, prompt: str):
        prompt_embeds, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, device=self.device,
            num_images_per_prompt=1, max_sequence_length=512,
        )
        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(self, prompt: str):
        """Returns (T5 embeds, pooled CLIP embeds). Requires text encoders still loaded."""
        if self.pipeline.text_encoder is None:
            raise RuntimeError("Text encoders have been freed. Call precompute_prompt() before initializing the sampler, "
                              "or set cached_prompt_embeds directly before sampling.")
        return self._encode_prompt_internal(prompt)

    def precompute_prompt(self, prompt: str):
        """Cache embeddings for `prompt`. Must be called before text encoders are freed."""
        if self.pipeline.text_encoder is not None:
            self.cached_prompt_embeds, self.cached_pooled_prompt_embeds = self._encode_prompt_internal(prompt)
        else:
            raise RuntimeError("Text encoders already deleted. Cannot precompute prompt.")

    def set_prompt_embeddings(self, prompt_key: str):
        """Switch to a prompt cached at init via `additional_prompts`."""
        if prompt_key not in self.prompt_cache:
            available = list(self.prompt_cache.keys())[:3]
            raise KeyError(f"Prompt not in cache: '{prompt_key[:50]}...'. Available: {available}")
        embeds, pooled = self.prompt_cache[prompt_key]
        self.cached_prompt_embeds = embeds.to(self.device)
        self.cached_pooled_prompt_embeds = pooled.to(self.device)
        self.prompt_text = prompt_key  # Update raw prompt for reward model tokenizers

    @torch.no_grad()
    def sample(self,
               latents: torch.Tensor,
               prompt_embeds: torch.Tensor,
               pooled_prompt_embeds: torch.Tensor,
               guidance_scale: float = 3.5,
               num_steps: int = 16) -> torch.Tensor:
        """Plain flow-ODE sampling on FLUX FlowMap; t goes from 1 (noise) to 0 (data)."""
        device = latents.device
        time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        z = latents
        for i in range(num_steps):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]
            u = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                    guidance_scale=guidance_scale, t_next=t_next.item())
            z = z - (t_cur - t_next) * u
        return z.to(torch.float32)

    @torch.no_grad()
    def sample_forward(self, num_steps: int = 4, height: int = 256, width: int = 256,
                       guidance_scale: float = 3.5, seed: int = 42,
                       initial_noise: torch.Tensor = None) -> tuple:
        """Unguided FLUX FlowMap sampling. Returns (PIL, unpacked latent, initial noise)."""
        batch_size = 1
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4  # 16

        H_latent = 2 * (int(height) // (self.vae_scale_factor * 2))
        W_latent = 2 * (int(width) // (self.vae_scale_factor * 2))

        if initial_noise is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            shape = (batch_size, num_channels_latents, H_latent, W_latent)
            latents_unpacked = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)
            initial_noise_unpacked = latents_unpacked.clone()
            latents = self.pipeline._pack_latents(latents_unpacked, batch_size, num_channels_latents, H_latent, W_latent)
        else:
            initial_noise_unpacked = initial_noise.clone()
            latents = self.pipeline._pack_latents(initial_noise, batch_size, num_channels_latents, H_latent, W_latent)

        # Prepare latent_image_ids
        latent_image_ids = self.pipeline._prepare_latent_image_ids(
            batch_size, H_latent // 2, W_latent // 2, self.device, self.dtype
        )

        # Prepare text_ids
        text_ids = torch.zeros(self.cached_prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)

        # Get timesteps using the same schedule as pipeline
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        image_seq_len = latents.shape[1]  # Packed sequence length

        mu = calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.get("base_image_seq_len", 256),
            self.pipeline.scheduler.config.get("max_image_seq_len", 4096),
            self.pipeline.scheduler.config.get("base_shift", 0.5),
            self.pipeline.scheduler.config.get("max_shift", 1.15),
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipeline.scheduler,
            num_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )

        # Reset scheduler state
        self.pipeline.scheduler.set_begin_index(0)

        print(f"Forward FlowMap sampling with {num_steps} steps...")

        for i, t in enumerate(tqdm(timesteps)):
            # Get t_next
            if i == len(timesteps) - 1:
                timestep2 = torch.zeros_like(t)
            else:
                timestep2 = timesteps[i + 1]

            # Predict velocity
            noise_pred = self.predict_velocity_packed(latents, t, timestep2, guidance_scale,
                                                       latent_image_ids=latent_image_ids,
                                                       text_ids=text_ids)

            # Scheduler step
            latents_dtype = latents.dtype
            latents = self.pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        # Unpack and decode
        latents_unpacked = self.pipeline._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents_scaled = (latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents_scaled, return_dict=False)[0]

        # Post-process using pipeline's image_processor for pixel-identical output
        image_pil = self.pipeline.image_processor.postprocess(image.detach(), output_type="pil")[0]

        return image_pil, latents_unpacked.detach(), initial_noise_unpacked

    def sample_reward_guided(self,
                            reward_type: str = "ensemble",
                            reward_model=None,
                            num_steps: int = 28,
                            guidance_scale: float = 3.5,
                            step_size: float = 0.1,
                            img_shape: Optional[Tuple[int, int]] = None,
                            latent: Optional[torch.Tensor] = None,
                            callback: Optional[callable] = None,
                            grad_mode: str = "jac",
                            sample_mode: str = "flow_map1",
                            seed: Optional[int] = None,
                            num_optim_iters: int = 3,
                            lmbda: float = 0.0,
                            use_adam: bool = True,
                            early_stop: int = 0,
                            noise_mode: str = "deterministic",
                            grad_norm_mode: str = "none",
                            enable_callback: bool = True,
                            grad_checkpointing: bool = False,
                            warmup_steps: int = 0,
                            warmup_particles: int = 1,
                            warmup_lr: Optional[float] = None,
                            unguided_steps: int = 0,
                            progress_lookahead_steps: int = 0,
                            force_particle_idx: Optional[int] = None,) -> torch.Tensor:
        """
        FLUX FlowMap reward-guided generation.

        Args:
            reward_type: kept as a string for forward-compat.
            reward_model: RewardEnsemble instance.
            num_steps: Number of sampling steps.
            guidance_scale: FLUX embedded CFG scale (distinct from FMRG λ).
            step_size: Reward guidance strength λ.
            img_shape: Output image shape (H, W).
            latent: Initial latent. If None, sample N(0, I).
            callback: Optional progress callback.
            grad_mode: "jac" (FMRG-J) or "euc" (FMRG-E).
            sample_mode: how the flow map is used to advance the trajectory.
                "flow_map2"    : canonical 2-NFE/step (X_{t,1} for both endpoint and step).
                "flow_map1"    : 1-NFE/step linear-interpolation shortcut.
                "flow_matching": 1-NFE Euler step (used by FlowDPS/FlowChef baselines).
            seed: Random seed.
            num_optim_iters: Inner gradient steps n_opt per interval.
            lmbda: Text-alignment regularization weight (default 0).
            use_adam: Adam optimizer with decaying LR.
            early_stop: Stop guidance at step N then complete with one uncontrolled
                flow-map step to t=0. 0 disables.
            noise_mode: "deterministic" or "stochastic" renoising.
            grad_norm_mode: "none" or "normalize" (velocity-norm rescaling).
            enable_callback: Enable per-step callback.
            grad_checkpointing: Enable transformer gradient checkpointing (saves VRAM).
            warmup_steps: Reinitialization rounds; 0 disables.
            warmup_particles: Particles per reinitialization.
            warmup_lr: Learning rate during reinitialization (default: step_size).
            unguided_steps: Number of trailing uncontrolled flow-map steps to t=0.
        Returns:
            Generated image tensor in [0, 1].
        """
        imgH, imgW = img_shape if img_shape is not None else (256, 256)

        prompt_embeds = self.cached_prompt_embeds
        pooled_prompt_embeds = self.cached_pooled_prompt_embeds

        if latent is None:
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            latent_shape = (1, 16, imgH // self.vae_scale_factor, imgW // self.vae_scale_factor)
            z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
        else:
            z = latent

        if grad_mode == "jac":
            z.requires_grad_(True)

        # Enable gradient checkpointing to reduce VRAM (recomputes activations during backward)
        if grad_checkpointing:
            self.pipeline.transformer.enable_gradient_checkpointing()

        device = z.device

        # Time schedule using FLUX pipeline scheduler
        latent_h = 2 * (int(imgH) // (self.vae_scale_factor * 2))
        latent_w = 2 * (int(imgW) // (self.vae_scale_factor * 2))
        image_seq_len = (latent_h // 2) * (latent_w // 2)

        mu = calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.get("base_image_seq_len", 256),
            self.pipeline.scheduler.config.get("max_image_seq_len", 4096),
            self.pipeline.scheduler.config.get("base_shift", 0.5),
            self.pipeline.scheduler.config.get("max_shift", 1.15),
        )

        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        timesteps, _ = retrieve_timesteps(
            self.pipeline.scheduler,
            num_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        time_steps = torch.cat([timesteps / 1000.0, torch.zeros(1, device=device)])

        n_keep_end = early_stop if (early_stop > 0 and early_stop < num_steps) else num_steps
        if n_keep_end < num_steps:
            time_steps = torch.cat([time_steps[:n_keep_end], torch.tensor([0.0], device=device)])
            actual_steps = n_keep_end
        else:
            actual_steps = num_steps

        # Warm-up: run first warmup_steps on N particles, pick best by reward
        start_step = 0
        best_z_before_last = None
        wu_steps = 0
        # Per-particle snapshots: wu_snapshots[p_idx][wi] = z after warmup step wi
        wu_snapshots = {}
        if warmup_steps > 0 and warmup_particles > 1 and step_size > 0 and reward_type == "ensemble":
            reward_ensemble = reward_model
            wu_steps = min(warmup_steps, actual_steps)
            print(f"Warm-up: {warmup_particles} particles × {wu_steps} guided steps")

            best_z = None
            best_z_before_last = None
            best_reward = float('-inf')

            for p_idx in range(warmup_particles):
                if seed is not None:
                    torch.manual_seed(seed + p_idx)
                    torch.cuda.manual_seed(seed + p_idx)
                z_p = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
                if grad_mode == "jac":
                    z_p.requires_grad_(True)

                z_p_before_last = None
                wu_snapshots[p_idx] = []

                # Run warmup_steps of the normal guidance loop
                for wi in range(wu_steps):
                    t_cur = time_steps[wi]
                    t_next = time_steps[wi + 1]
                    dt = t_cur - t_next

                    # Save state before last warmup step (for unguided_steps)
                    if unguided_steps > 1 and wi == wu_steps - 1:
                        z_p_before_last = z_p.detach().clone()

                    if grad_mode == "jac" and not z_p.requires_grad:
                        z_p = z_p.detach().requires_grad_(True)

                    if grad_mode == "jac":
                        u = self.predict_vector(z_p, t_cur.item(), prompt_embeds=prompt_embeds,
                                                pooled_prompt_embeds=pooled_prompt_embeds,
                                                guidance_scale=guidance_scale, t_next=0.0)
                    else:
                        with torch.no_grad():
                            u = self.predict_vector(z_p, t_cur.item(), prompt_embeds=prompt_embeds,
                                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                                    guidance_scale=guidance_scale, t_next=0.0)

                    z0t = z_p - t_cur * u
                    u_step = u if sample_mode == "flow_map1" else u
                    if sample_mode == "flow_map2":
                        with torch.no_grad():
                            u_step = self.predict_vector(z_p, t_cur.item(), prompt_embeds=prompt_embeds,
                                                         pooled_prompt_embeds=pooled_prompt_embeds,
                                                         guidance_scale=guidance_scale, t_next=t_next.item())

                    if grad_mode == "euc":
                        z0t_opt, _, _ = self.reward_consistency(
                            z0t=z0t, reward_ensemble=reward_ensemble, prompt=self.prompt_text,
                            stepsize=step_size, num_iters=num_optim_iters,
                            lmbda=lmbda, timestep_idx=wi, total_timesteps=actual_steps, use_adam=use_adam)
                        grad_z0 = -(z0t_opt - z0t).to(self.dtype)
                        wt = t_cur * (1 - t_next)
                        z_p = z_p - dt * u_step - wt * grad_z0

                    elif grad_mode == "jac":
                        vel_norm = torch.linalg.vector_norm(u_step, dim=(1, 2, 3), keepdim=True).detach() if grad_norm_mode in ["clip", "normalize"] else None
                        vel_t_next = 0.0 if sample_mode in ["flow_map1", "flow_map2"] else t_next.item()
                        grad_zt, _, zt_original = self.reward_consistency_xt(
                            zt=z_p, t_cur=t_cur.item(), reward_ensemble=reward_ensemble,
                            prompt=self.prompt_text, prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds, guidance_scale=guidance_scale,
                            stepsize=step_size, num_iters=num_optim_iters, lmbda=lmbda,
                            grad_norm_mode=grad_norm_mode, velocity_norm=vel_norm,
                            velocity=u, t_next=vel_t_next,
                            use_adam=use_adam, timestep_idx=wi, total_timesteps=actual_steps)
                        guided_velocity = u_step.detach() + grad_zt
                        z_p = zt_original - dt * guided_velocity
                        z_p = z_p.detach().requires_grad_(True)

                    if z_p.dtype != self.dtype:
                        z_p = z_p.to(self.dtype)

                    # Save snapshot of this particle's z after the warmup step (for callback replay).
                    if enable_callback and callback is not None:
                        wu_snapshots[p_idx].append(z_p.detach().clone())

                # Evaluate reward at warmup end via z0t lookahead
                with torch.no_grad():
                    u_eval = self.predict_vector(z_p.detach(), time_steps[wu_steps].item(),
                                                 prompt_embeds=prompt_embeds,
                                                 pooled_prompt_embeds=pooled_prompt_embeds,
                                                 guidance_scale=guidance_scale, t_next=0.0)
                    z0_eval = z_p.detach() - time_steps[wu_steps] * u_eval
                    img_eval = self.decode(z0_eval).float()
                    img_01 = ((img_eval + 1) / 2).clamp(0, 1)
                    reward_loss, _ = reward_ensemble(img_01, self.prompt_text)
                    r_val = -reward_loss.item()

                print(f"  Particle {p_idx}: reward={r_val:.4f}")
                # Always save this particle's final state so force_particle_idx can override later.
                if not hasattr(self, "_all_particle_z"):
                    self._all_particle_z = {}
                self._all_particle_z[p_idx] = (z_p.detach().clone(), z_p_before_last)
                if r_val > best_reward:
                    best_reward = r_val
                    best_z = z_p.detach().clone()
                    best_z_before_last = z_p_before_last
                    best_p_idx = p_idx

            # Override selection if force_particle_idx is provided.
            if force_particle_idx is not None and force_particle_idx in self._all_particle_z:
                forced_z, forced_zbl = self._all_particle_z[force_particle_idx]
                best_z = forced_z
                best_z_before_last = forced_zbl
                best_p_idx = force_particle_idx
                print(f"  Forcing particle {force_particle_idx} (overriding reward-based selection)")

            z = best_z.to(self.dtype)
            if grad_mode == "jac":
                z.requires_grad_(True)
            start_step = wu_steps
            self._selected_particle_idx = best_p_idx
            print(f"  Selected particle {best_p_idx}, continuing from step {start_step}")

            # Replay the chosen particle's warmup snapshots through the callback so progress
            # images cover the warmup phase (frames 1..wu_steps).
            if enable_callback and callback is not None and best_p_idx in wu_snapshots:
                for wi, z_snap in enumerate(wu_snapshots[best_p_idx]):
                    t_at_step = float(time_steps[wi + 1].item())
                    with torch.no_grad():
                        if progress_lookahead_steps > 0:
                            img_progress = self._progress_lookahead_decode(
                                z_snap, t_at_step, int(progress_lookahead_steps),
                                imgH, imgW, guidance_scale)
                        else:
                            img_progress = self.decode(z_snap)
                    callback(wi + 1, img_progress, t_at_step, 0.0)

        pbar = tqdm(range(start_step, actual_steps), total=actual_steps - start_step, desc='FLUX-FMRG')

        z_before_last = None
        t_before_last = None

        for i in pbar:
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]
            dt = t_cur - t_next

            # Save state before last step (for unguided_steps)
            if unguided_steps > 1 and i == actual_steps - 1:
                z_before_last = z.detach().clone()
                t_before_last = t_cur.item()

            # Enable gradients on z if needed for xt mode
            if grad_mode == "jac" and not z.requires_grad:
                z = z.detach().requires_grad_(True)
            elif grad_mode != "jac" and z.requires_grad:
                z = z.detach()

            reward_loss = 0.0

            if step_size == 0:
                # No guidance - standard Euler step
                with torch.no_grad():
                    if sample_mode == "flow_map1":
                        u = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                                pooled_prompt_embeds=pooled_prompt_embeds,
                                                guidance_scale=guidance_scale, t_next=0.0)
                    elif sample_mode == "flow_map2":
                        u = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                                pooled_prompt_embeds=pooled_prompt_embeds,
                                                guidance_scale=guidance_scale, t_next=t_next.item())
                    else:
                        # flow_matching: instantaneous velocity
                        u = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                                pooled_prompt_embeds=pooled_prompt_embeds,
                                                guidance_scale=guidance_scale, t_next=t_cur.item())
                z0t = z - t_cur * u
                z = z - dt * u
            else:
                # xt mode needs gradients through u.
                if grad_mode == "jac":
                    if sample_mode in ["flow_map1", "flow_map2"]:
                        u = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                                pooled_prompt_embeds=pooled_prompt_embeds,
                                                guidance_scale=guidance_scale, t_next=0.0)
                    else:
                        u = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                                pooled_prompt_embeds=pooled_prompt_embeds,
                                                guidance_scale=guidance_scale, t_next=t_cur.item())
                else:
                    with torch.no_grad():
                        if sample_mode in ["flow_map1", "flow_map2"]:
                            u = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                                    guidance_scale=guidance_scale, t_next=0.0)
                        else:
                            u = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                                    guidance_scale=guidance_scale, t_next=t_cur.item())

                z0t = z - t_cur * u

                # Determine u_step for Euler stepping
                if sample_mode == "flow_map1":
                    u_step = u
                elif sample_mode == "flow_map2":
                    with torch.no_grad():
                        u_step = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                                     pooled_prompt_embeds=pooled_prompt_embeds,
                                                     guidance_scale=guidance_scale, t_next=t_next.item())
                else:
                    u_step = u

                reward_ensemble = reward_model

                if grad_mode == "euc":
                    # FMRG-E (Euclidean): wt = t * (1 - t_next).
                    z0t_opt, reward_loss, _ = self.reward_consistency(
                        z0t=z0t, reward_ensemble=reward_ensemble,
                        prompt=self.prompt_text,
                        stepsize=step_size, num_iters=num_optim_iters,
                        lmbda=lmbda, timestep_idx=i, total_timesteps=actual_steps,
                        use_adam=use_adam)

                    grad_z0 = -(z0t_opt - z0t).to(self.dtype)
                    wt = t_cur * (1 - t_next)
                    z = z - dt * u_step - wt * grad_z0

                elif grad_mode == "jac":
                    # Gradient w.r.t z_t
                    vel_norm = torch.linalg.vector_norm(u_step, dim=(1, 2, 3), keepdim=True).detach() if grad_norm_mode in ["clip", "normalize"] else None
                    vel_t_next = 0.0 if sample_mode in ["flow_map1", "flow_map2"] else t_next.item()

                    grad_zt, reward_loss, zt_original = self.reward_consistency_xt(
                        zt=z, t_cur=t_cur.item(),
                        reward_ensemble=reward_ensemble,
                        prompt=self.prompt_text,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        guidance_scale=guidance_scale,
                        stepsize=step_size, num_iters=num_optim_iters,
                        lmbda=lmbda,
                        grad_norm_mode=grad_norm_mode, velocity_norm=vel_norm,
                        velocity=u, t_next=vel_t_next,
                        use_adam=use_adam, timestep_idx=i, total_timesteps=actual_steps)

                    guided_velocity = u_step.detach() + grad_zt
                    if noise_mode == "stochastic":
                        # Decompose guided update into z0/z1 endpoints, mix noise, recompose at t_next.
                        z0_guided = zt_original - t_cur * guided_velocity
                        z1_guided = zt_original + (1 - t_cur) * guided_velocity
                        noise = math.sqrt(t_next) * z1_guided + math.sqrt(1 - t_next) * torch.randn_like(z1_guided)
                        z = z0_guided + t_next * (noise - z0_guided)
                    else:
                        z = zt_original - dt * guided_velocity
                    z = z.detach()
                    z.requires_grad_(True)

                else:
                    raise ValueError(f"Unknown grad_mode: {grad_mode}")

            if z.dtype != self.dtype:
                z = z.to(self.dtype)

            # Callback for progress tracking
            if enable_callback and callback is not None:
                with torch.no_grad():
                    if progress_lookahead_steps > 0:
                        img_progress = self._progress_lookahead_decode(
                            z, float(t_next.item()), int(progress_lookahead_steps),
                            imgH, imgW, guidance_scale)
                    else:
                        img_progress = self.decode(z0t)
                callback(i + 1, img_progress, t_next.item(), reward_loss)

            pbar.set_postfix({'loss': f'{reward_loss:.2f}'})

        # Disable gradient checkpointing (restore normal inference speed)
        if grad_checkpointing:
            self.pipeline.transformer.disable_gradient_checkpointing()

        # Final decode
        with torch.no_grad():
            if early_stop > 0:
                # Fall back to warmup's saved pre-jump state if main loop didn't run
                if z_before_last is None and best_z_before_last is not None:
                    z_before_last = best_z_before_last.to(self.dtype)
                    t_before_last = time_steps[wu_steps - 1].item() if wu_steps > 0 else None
                if unguided_steps > 1 and z_before_last is not None:
                    # Redo the last jump as multiple flow map steps
                    z = z_before_last
                    t_remain = t_before_last
                    dt_ug = t_remain / unguided_steps
                    for ug_i in range(unguided_steps - 1):
                        t_cur_ug = t_remain - ug_i * dt_ug
                        t_next_ug = t_cur_ug - dt_ug
                        u_ug = self.predict_vector(z, t_cur_ug, prompt_embeds=prompt_embeds,
                                                    pooled_prompt_embeds=pooled_prompt_embeds,
                                                    guidance_scale=guidance_scale, t_next=t_next_ug)
                        z = z - dt_ug * u_ug
                    t_remain = dt_ug
                else:
                    t_remain = t_next.item()
                # Final flow map jump to t=0
                u_final = self.predict_vector(z, t_remain, prompt_embeds=prompt_embeds,
                                               pooled_prompt_embeds=pooled_prompt_embeds,
                                               guidance_scale=guidance_scale, t_next=0.0)
                z0_final = z - t_remain * u_final
                img = self.decode(z0_final)
            else:
                img = self.decode(z)

        return img

