"""FLUX FlowMap sampler for inverse problems."""

import os
import torch
import gc
import math
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple
import sys

# =============================================================================
# CONFIGURABLE PATHS - Modify these for your setup
# =============================================================================

# Directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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


class FluxFlowMapSampler:
    """
    FLUX FlowMap sampler with integrated VAE and model loading
    Uses FluxPipelineTwoTimestep for dual timestep conditioning
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, lora_path: str = DEFAULT_LORA_PATH,
                 device='cuda', dtype=torch.bfloat16, prompt: str = ""):
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.lora_path = lora_path

        torch.backends.cuda.matmul.allow_tf32 = True

        self._load_model(prompt=prompt)

        self.vae = self.pipeline.vae
        self.vae_scale_factor = self.pipeline.vae_scale_factor  # 16 for FLUX

    def _load_model(self, prompt: str = ""):
        print(f"Loading FLUX FlowMap pipeline from: {self.model_id}")
        print(f"Loading LoRA weights from: {self.lora_path}")

        self.pipeline = FluxPipelineTwoTimestep.from_pretrained(
            self.model_id, torch_dtype=self.dtype, cache_dir=HF_CACHE,
        ).to(self.device)

        self.pipeline.transformer = add_dual_time_embedder(self.pipeline.transformer)
        self.pipeline.load_lora_weights(self.lora_path, weight_name="pytorch_lora_weights.safetensors")

        print("Precomputing text embeddings for memory efficiency...")
        with torch.no_grad():
            self.null_prompt_embeds, self.null_pooled_prompt_embeds = self._encode_prompt_internal("")

            if isinstance(prompt, list):
                unique_prompts = list(dict.fromkeys(prompt))  # deduplicate, preserve order
                self.prompt_embeddings = {}
                for p in unique_prompts:
                    pe, ppe = self._encode_prompt_internal(p if p else "")
                    self.prompt_embeddings[p] = (pe, ppe)
                self.cached_prompt_embeds = self.prompt_embeddings[unique_prompts[0]][0]
                self.cached_pooled_prompt_embeds = self.prompt_embeddings[unique_prompts[0]][1]
                print(f"Precomputed embeddings for {len(unique_prompts)} unique prompts")
            elif prompt:
                self.cached_prompt_embeds, self.cached_pooled_prompt_embeds = self._encode_prompt_internal(prompt)
                self.prompt_embeddings = {prompt: (self.cached_prompt_embeds, self.cached_pooled_prompt_embeds)}
                print(f"Precomputed embeddings for prompt: '{prompt[:50]}...'")
            else:
                self.cached_prompt_embeds = self.null_prompt_embeds.clone()
                self.cached_pooled_prompt_embeds = self.null_pooled_prompt_embeds.clone()
                self.prompt_embeddings = {"": (self.cached_prompt_embeds, self.cached_pooled_prompt_embeds)}

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

    def set_prompt_embeddings(self, prompt: str):
        """Switch the active prompt embeddings (for class-conditional generation)."""
        if prompt in self.prompt_embeddings:
            self.cached_prompt_embeds, self.cached_pooled_prompt_embeds = self.prompt_embeddings[prompt]
        else:
            raise ValueError(f"Prompt not precomputed: '{prompt[:50]}...'. "
                             f"Available: {list(self.prompt_embeddings.keys())[:5]}")

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

    def _compute_residual_loss(self, residual: torch.Tensor, loss_func: str = "sum") -> torch.Tensor:
        """Per-batch-item reduction over the residual, then summed across the batch
        so each item's gradient stays independent."""
        batch_size = residual.shape[0]
        residual_flat = residual.reshape(batch_size, -1)
        if loss_func == "norm":
            return torch.linalg.norm(residual_flat, dim=1).sum()
        return (residual_flat ** 2).sum(dim=1).sum()

    def compute_loss(self, img, operator, measurement, task, loss_mode: str = "pixel", loss_func: str = "sum"):
        """Compute measurement loss for a decoded image (no gradients)."""
        x = img.float()
        if loss_mode == "pixel":
            residual = measurement - operator.A(x)
        else:
            if "sr" in task:
                residual = operator.A_pinv(measurement) - operator.A_pinv(operator.A(x))
            else:
                residual = operator.At(measurement) - operator.At(operator.A(x))
        loss = self._compute_residual_loss(residual, loss_func)
        return loss.item()

    def fmrg_gradient_step_euclidean(self, z0t, operator, measurement, task, stepsize: float = 30.0, num_iters: int = 3,
                                     loss_mode: str = "pixel", loss_func: str = "sum",
                                     normalize_grad: bool = False, velocity_norm: torch.Tensor = None):
        """
        FMRG-E gradient step on z0t (clean-latent prediction): iteratively
        minimizes the measurement loss with respect to z0t, optionally rescaling
        the gradient to the velocity norm.
        """
        z0t_opt = z0t.clone().requires_grad_(True)
        data_loss = None
        for _ in range(num_iters):
            x0t = self.decode(z0t_opt).float()
            if loss_mode == "pixel":
                residual = measurement - operator.A(x0t)
                data_loss = self._compute_residual_loss(residual, loss_func)
            else:  # latent
                if "sr" in task:
                    residual = operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))
                else:
                    residual = operator.At(measurement) - operator.At(operator.A(x0t))
                data_loss = self._compute_residual_loss(residual, loss_func)

            grad = torch.autograd.grad(data_loss, z0t_opt)[0].to(self.dtype)

            if normalize_grad:
                grad_norm = torch.linalg.vector_norm(grad, dim=(1, 2, 3), keepdim=True)
                grad = grad / (grad_norm + 1e-8)
                if velocity_norm is not None:
                    grad = grad * velocity_norm

            z0t_opt = (z0t_opt - stepsize * grad).detach().requires_grad_(True)

        return z0t_opt.detach(), data_loss.item() if data_loss is not None else 0.0

    def fmrg_gradient_step_jacobian(self, zt, t_cur, operator, measurement, task,
                                    prompt_embeds=None, pooled_prompt_embeds=None,
                                    guidance_scale: float = 3.5,
                                    stepsize: float = 30.0, num_iters: int = 3,
                                    loss_mode: str = "pixel",
                                    loss_func: str = "sum", velocity: torch.Tensor = None,
                                    normalize_grad: bool = False, velocity_norm: torch.Tensor = None,
                                    t_next: float = None):
        """
        FMRG-J gradient step on z_t (noisy latent): backpropagates the
        measurement loss through the flow-map endpoint X_{t,1}(z_t), optionally
        rescaling the per-iteration gradient to the velocity norm.
        """
        if t_next is None:
            raise ValueError("t_next must be provided for fmrg_gradient_step_jacobian")

        zt_original = zt.clone().detach()
        data_loss = None

        for iter_idx in range(num_iters):
            if iter_idx == 0 and velocity is not None:
                u = velocity
            else:
                u = self.predict_vector(zt, t_cur, prompt_embeds=prompt_embeds,
                                        pooled_prompt_embeds=pooled_prompt_embeds,
                                        guidance_scale=guidance_scale, t_next=None)

            z0t = zt - t_cur * u
            x0t = self.decode(z0t).float()

            if loss_mode == "pixel":
                residual = measurement - operator.A(x0t)
            elif "sr" in task:
                residual = operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))
            else:
                residual = operator.At(measurement) - operator.At(operator.A(x0t))
            data_loss = self._compute_residual_loss(residual, loss_func)

            grad = torch.autograd.grad(data_loss, zt)[0]

            if normalize_grad:
                grad_norm = torch.linalg.vector_norm(grad, dim=(1, 2, 3), keepdim=True)
                grad = grad / (grad_norm + 1e-8)
                if velocity_norm is not None:
                    grad = grad * velocity_norm

            zt = (zt - stepsize * grad).detach().requires_grad_(True)

        grad_zt = -(zt.detach() - zt_original)
        return grad_zt, data_loss.item() if data_loss is not None else 0.0, zt_original

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

        timestep = t_cur.expand(batch_size).to(z_packed.dtype)
        timestep2 = t_next.expand(batch_size).to(z_packed.dtype)

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

            # Update z: z_next = z - (t_cur - t_next) * u
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

        latent_image_ids = self.pipeline._prepare_latent_image_ids(
            batch_size, H_latent // 2, W_latent // 2, self.device, self.dtype
        )
        text_ids = torch.zeros(self.cached_prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)

        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.get("base_image_seq_len", 256),
            self.pipeline.scheduler.config.get("max_image_seq_len", 4096),
            self.pipeline.scheduler.config.get("base_shift", 0.5),
            self.pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            self.pipeline.scheduler, num_steps, self.device, sigmas=sigmas, mu=mu,
        )
        self.pipeline.scheduler.set_begin_index(0)

        print(f"Forward FlowMap sampling with {num_steps} steps...")
        for i, t in enumerate(tqdm(timesteps)):
            timestep2 = torch.zeros_like(t) if i == len(timesteps) - 1 else timesteps[i + 1]
            noise_pred = self.predict_velocity_packed(latents, t, timestep2, guidance_scale,
                                                     latent_image_ids=latent_image_ids,
                                                     text_ids=text_ids)
            latents_dtype = latents.dtype
            latents = self.pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        latents_unpacked = self.pipeline._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents_scaled = (latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents_scaled, return_dict=False)[0]
        image_pil = self.pipeline.image_processor.postprocess(image.detach(), output_type="pil")[0]

        return image_pil, latents_unpacked.detach(), initial_noise_unpacked

    def sample_inverse_problem(self,
                               measurement: torch.Tensor,
                               operator,
                               task: str,
                               num_steps: int = 28,
                               guidance_scale: float = 3.5,
                               step_size: float = 7.0,
                               img_shape: Optional[Tuple[int, int]] = None,
                               latent: Optional[torch.Tensor] = None,
                               callback: Optional[callable] = None,
                               grad_mode: str = "jac",
                               sample_mode: str = "flow_map1",
                               seed: Optional[int] = None,
                               num_optim_iters: int = 10,
                               loss_mode: str = "latent",
                               loss_func: str = "sum",
                               normalize_grad: bool = False,
                               early_stop: int = 0,
                               enable_callback: bool = True) -> torch.Tensor:
        """
        FLUX FlowMap FMRG sampling for inverse problems.

        Args:
            measurement: Observed measurement y.
            operator: Forward measurement operator A (from functions/measurements.py).
            task: Task name (e.g. "sr", "inpaint", "deblur").
            num_steps: Number of time-grid steps N.
            guidance_scale: FLUX embedded classifier-free guidance.
            step_size: Guidance strength λ.
            img_shape: Output image shape (H, W).
            latent: Initial latent z_0; random if None.
            grad_mode: "jac" (FMRG-J, gradient through the flow map (Jacobian)) or
                "euc" (FMRG-E, drops the Jacobian (Euclidean)).
            sample_mode: How the flow map advances the trajectory.
                "flow_map2": 2-NFE/step.
                "flow_map1": 1-NFE/step via linear interpolation.
                "flow_matching": single Euler step (used by baselines).
            seed: Random seed.
            num_optim_iters: Inner gradient steps per interval.
            loss_mode: "latent" or "pixel" reward computation space.
            loss_func: Reduction over the residual ("sum" or "norm").
            normalize_grad: Rescale the per-iteration gradient to the velocity norm
               .
            early_stop: Number of leading guided steps; 0 disables (full schedule).
                When > 0, completes with one uncontrolled flow-map step to t=0.
            enable_callback: Enable per-step callback (slower; decodes each step).
        """
        imgH, imgW = img_shape if img_shape is not None else (256, 256)
        batch_size = measurement.shape[0]

        prompt_embeds = self.cached_prompt_embeds
        pooled_prompt_embeds = self.cached_pooled_prompt_embeds
        if prompt_embeds.shape[0] != batch_size:
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.expand(batch_size, -1)

        if latent is None:
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            latent_shape = (batch_size, 16, imgH // self.vae_scale_factor, imgW // self.vae_scale_factor)
            z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
        else:
            z = latent

        if grad_mode == "jac":
            z.requires_grad_(True)

        device = z.device

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

        if early_stop > 0:
            n_keep_end = early_stop
            schedule_slice = time_steps[:n_keep_end]
            time_steps = torch.cat([schedule_slice, torch.zeros(1, device=device)])
            actual_steps = n_keep_end
        else:
            actual_steps = num_steps

        mode_desc = 'FlowMap' if sample_mode in ['flow_map1', 'flow_map2'] else 'FlowMatching'
        pbar = tqdm(range(actual_steps), total=actual_steps, desc=f'FLUX-{mode_desc}')

        for i in pbar:
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]
            dt = t_cur - t_next

            # FMRG-J keeps z under requires_grad; FMRG-E detaches.
            if grad_mode == "jac" and not z.requires_grad:
                z = z.detach().requires_grad_(True)
            elif grad_mode != "jac" and z.requires_grad:
                z = z.detach()

            dc_loss = 0.0

            if step_size == 0:
                with torch.no_grad():
                    t_next_arg = 0.0 if sample_mode in ["flow_map1", "flow_map2"] else t_cur.item()
                    u = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                            pooled_prompt_embeds=pooled_prompt_embeds,
                                            guidance_scale=guidance_scale, t_next=t_next_arg)
                z0t = z - t_cur * u
                z = z - dt * u
            else:
                if grad_mode == "jac":
                    if sample_mode in ["flow_map1", "flow_map2"]:
                        u = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                                pooled_prompt_embeds=pooled_prompt_embeds,
                                                guidance_scale=guidance_scale, t_next=0.0)
                    else:
                        # flow_matching: instantaneous velocity (t_next=t_cur)
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
                    u_step = u  # reuse the mean velocity computed with t_next=0
                elif sample_mode == "flow_map2":
                    with torch.no_grad():
                        u_step = self.predict_vector(z, t_cur.item(), prompt_embeds=prompt_embeds,
                                                     pooled_prompt_embeds=pooled_prompt_embeds,
                                                     guidance_scale=guidance_scale, t_next=t_next.item())
                else:  # flow_matching
                    u_step = u

                if grad_mode == "euc":
                    # FMRG-E (Euclidean): direct correction applied to the x0 prediction.
                    vel_norm = torch.linalg.vector_norm(u_step, dim=(1, 2, 3), keepdim=True).detach() if normalize_grad else None

                    z0t_opt, dc_loss = self.fmrg_gradient_step_euclidean(
                        z0t, operator, measurement, task=task,
                        stepsize=step_size, num_iters=num_optim_iters,
                        loss_mode=loss_mode, loss_func=loss_func,
                        normalize_grad=normalize_grad, velocity_norm=vel_norm,
                    )

                    grad_z0 = -(z0t_opt - z0t)
                    wt = t_cur * (1 - t_next)
                    if normalize_grad:
                        z = z - dt * u_step - dt * wt * grad_z0
                    else:
                        z = z - dt * u_step - wt * grad_z0

                elif grad_mode == "jac":
                    # FMRG-J (Jacobian): gradient through the flow map / velocity network.
                    vel_norm = torch.linalg.vector_norm(u_step, dim=(1, 2, 3), keepdim=True).detach() if normalize_grad else None
                    vel_t_next = 0.0 if sample_mode in ["flow_map1", "flow_map2"] else t_next.item()

                    grad_zt, dc_loss, zt_original = self.fmrg_gradient_step_jacobian(
                        z, t_cur.item(), operator, measurement, task=task,
                        prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                        guidance_scale=guidance_scale,
                        stepsize=step_size, num_iters=num_optim_iters,
                        loss_mode=loss_mode, loss_func=loss_func, velocity=u,
                        normalize_grad=normalize_grad, velocity_norm=vel_norm,
                        t_next=vel_t_next,
                    )

                    guided_velocity = u_step.detach() + grad_zt
                    z = zt_original - dt * guided_velocity
                    z = z.detach().requires_grad_(True)
                else:
                    raise ValueError(f"Unknown grad_mode: {grad_mode!r} (expected 'jac' or 'euc')")

            if enable_callback and callback is not None:
                with torch.no_grad():
                    z0_decoded = self.decode(z0t)
                callback(i, z0_decoded, t_cur.item(), dc_loss)

        # Decode final latent
        with torch.no_grad():
            img = self.decode(z)

        if enable_callback and callback is not None:
            with torch.no_grad():
                final_loss = self.compute_loss(img, operator, measurement, task=task, loss_mode=loss_mode, loss_func=loss_func)
            callback(actual_steps, img, 0.0, final_loss)

        return img


class FluxFlowDPS(FluxFlowMapSampler):
    """FlowDPS baseline ported to FLUX FlowMap."""
    def data_consistency_flowdps(self, z0t, operator, measurement, task, stepsize: float = 30.0, num_iters: int = 3):
        """
        Iterative optimization of z0t under the measurement loss. Computes
        per-batch-item loss to keep gradients independent across the batch.
        """
        batch_size = z0t.shape[0]
        z0t = z0t.clone().detach().requires_grad_(True)
        for _ in range(num_iters):
            x0t = self.decode(z0t).float()
            if "sr" in task:
                residual = operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))
            else:
                residual = operator.At(measurement) - operator.At(operator.A(x0t))
            residual_flat = residual.view(batch_size, -1)
            loss = torch.linalg.norm(residual_flat, dim=1).sum()
            grad = torch.autograd.grad(loss, z0t)[0].to(self.dtype)
            z0t = (z0t - stepsize * grad).detach().requires_grad_(True)

        return z0t.detach()

    def sample_flowdps(self, measurement, operator, task,
                       NFE: int = 28,
                       img_shape: Optional[Tuple[int, int]] = None,
                       guidance_scale: float = 3.5,
                       step_size: float = 30.0,
                       num_optim_iters: int = 3,
                       latent: Optional[torch.Tensor] = None,
                       callback: Optional[callable] = None,
                       seed: Optional[int] = None):
        """FlowDPS baseline sampler. Uses instantaneous velocity (flow-matching mode)."""
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)
        batch_size = measurement.shape[0]

        prompt_embeds = self.cached_prompt_embeds
        pooled_prompt_embeds = self.cached_pooled_prompt_embeds
        if prompt_embeds.shape[0] != batch_size:
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.expand(batch_size, -1)

        if latent is None:
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            latent_shape = (batch_size, 16, imgH // self.vae_scale_factor, imgW // self.vae_scale_factor)
            z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
        else:
            z = latent

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

        sigmas_np = np.linspace(1.0, 1 / NFE, NFE)
        timesteps, _ = retrieve_timesteps(
            self.pipeline.scheduler, NFE, self.device, sigmas=sigmas_np, mu=mu,
        )
        sigmas = torch.cat([timesteps / 1000.0, torch.zeros(1, device=self.device)])

        pbar = tqdm(range(NFE), total=NFE, desc='FLUX-FlowDPS')
        for i in pbar:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            z = z.detach()

            with torch.no_grad():
                pred_v = self.predict_vector(z, sigma.item(), prompt_embeds=prompt_embeds,
                                             pooled_prompt_embeds=pooled_prompt_embeds,
                                             guidance_scale=guidance_scale, t_next=sigma.item())

            z0t = z - sigma * pred_v
            z1t = z + (1 - sigma) * pred_v
            delta = sigma - sigma_next

            z0y = self.data_consistency_flowdps(z0t, operator, measurement, task=task,
                                                stepsize=step_size, num_iters=num_optim_iters)
            z0y = (1 - sigma) * z0t + sigma * z0y

            noise = math.sqrt(sigma_next) * z1t + math.sqrt(1 - sigma_next) * torch.randn_like(z1t)
            z = z0y + (sigma - delta) * (noise - z0y)

            if callback is not None:
                with torch.no_grad():
                    z0_decoded = self.decode(z0t)
                    dc_loss = self.compute_loss(z0_decoded, operator, measurement, task=task, loss_mode="pixel")
                callback(i, z0_decoded, sigma.item(), dc_loss)

        with torch.no_grad():
            img = self.decode(z)
        return img


class FluxFlowChef(FluxFlowMapSampler):
    """FlowChef baseline ported to FLUX FlowMap."""
    def data_consistency_flowchef(self, z0t, operator, measurement, task, stepsize: float = 30.0, num_iters: int = 1):
        """
        Single-step (num_iters=1) gradient correction on the x0 prediction; setting
        num_iters > 1 iterates the same update.

        Returns the accumulated update applied to z0t over all iterations.
        """
        batch_size = z0t.shape[0]
        z0t = z0t.clone().detach().requires_grad_(True)
        total_update = torch.zeros_like(z0t)

        for _ in range(num_iters):
            x0t = self.decode(z0t).float()
            if "sr" in task:
                residual = operator.A_pinv(measurement) - operator.A_pinv(operator.A(x0t))
            else:
                residual = operator.At(measurement) - operator.At(operator.A(x0t))
            residual_flat = residual.view(batch_size, -1)
            loss = torch.linalg.norm(residual_flat, dim=1).sum()
            grad = torch.autograd.grad(loss, z0t)[0].to(self.dtype)
            total_update = total_update + stepsize * grad
            z0t = (z0t - stepsize * grad).detach().requires_grad_(True)

        return total_update.detach()

    def sample_flowchef(self, measurement, operator, task,
                        NFE: int = 28,
                        img_shape: Optional[Tuple[int, int]] = None,
                        guidance_scale: float = 3.5,
                        step_size: float = 50.0,
                        num_optim_iters: int = 1,
                        latent: Optional[torch.Tensor] = None,
                        callback: Optional[callable] = None,
                        seed: Optional[int] = None):
        """FlowChef baseline sampler. Uses instantaneous velocity (flow-matching mode)."""
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)
        batch_size = measurement.shape[0]

        prompt_embeds = self.cached_prompt_embeds
        pooled_prompt_embeds = self.cached_pooled_prompt_embeds
        if prompt_embeds.shape[0] != batch_size:
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.expand(batch_size, -1)

        if latent is None:
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            latent_shape = (batch_size, 16, imgH // self.vae_scale_factor, imgW // self.vae_scale_factor)
            z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
        else:
            z = latent

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

        sigmas_np = np.linspace(1.0, 1 / NFE, NFE)
        timesteps, _ = retrieve_timesteps(
            self.pipeline.scheduler, NFE, self.device, sigmas=sigmas_np, mu=mu,
        )
        sigmas = torch.cat([timesteps / 1000.0, torch.zeros(1, device=self.device)])

        pbar = tqdm(range(NFE), total=NFE, desc='FLUX-FlowChef')
        for i in pbar:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            z = z.detach()

            with torch.no_grad():
                pred_v = self.predict_vector(z, sigma.item(), prompt_embeds=prompt_embeds,
                                             pooled_prompt_embeds=pooled_prompt_embeds,
                                             guidance_scale=guidance_scale, t_next=sigma.item())

            z0t = z - sigma * pred_v
            z1t = z + (1 - sigma) * pred_v
            delta = sigma - sigma_next

            grad = self.data_consistency_flowchef(z0t, operator, measurement, task=task,
                                                  stepsize=step_size, num_iters=num_optim_iters)

            z = z0t + (sigma - delta) * (z1t - z0t) - grad

            if callback is not None:
                with torch.no_grad():
                    z0_decoded = self.decode(z0t)
                    dc_loss = self.compute_loss(z0_decoded, operator, measurement, task=task, loss_mode="pixel")
                callback(i, z0_decoded, sigma.item(), dc_loss)

        with torch.no_grad():
            img = self.decode(z)
        return img


def create_fluxfm_sampler(model_id: str = DEFAULT_MODEL_ID, lora_path: str = DEFAULT_LORA_PATH, device='cuda', prompt: str = ""):
    """Factory function to create FLUX FlowMap sampler"""
    return FluxFlowMapSampler(model_id=model_id, lora_path=lora_path, device=device, prompt=prompt)

