"""FLUX FlowMap wrapper implementing the apply() interface for ReNO."""

import os
import sys
import gc
import torch
import numpy as np
from typing import List, Optional, Dict, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, REPO_ROOT)


class RewardFluxFlowMapPipeline:
    def __init__(self, resolution=512, device='cuda', dtype=torch.bfloat16,
                 guidance_scale=3.5, lora_path=None):
        self.resolution = resolution
        self.device = device
        self.dtype = dtype
        self.guidance_scale = guidance_scale

        if lora_path is None:
            env_var = "FMRG_LORA_PATH_512" if resolution >= 512 else "FMRG_LORA_PATH_256"
            lora_path = os.environ.get(env_var) or os.environ.get("FMRG_LORA_PATH")
            if lora_path is None:
                default = "checkpoints/flux-flowmap-lora-512" if resolution >= 512 else "checkpoints/flux-flowmap-lora"
                lora_path = os.path.join(REPO_ROOT, default)

        # Import and create sampler — do NOT pass prompt yet, we'll encode later
        from fluxfm_sampler_reward import FluxFlowMapSampler, DEFAULT_MODEL_ID
        self.sampler = FluxFlowMapSampler.__new__(FluxFlowMapSampler)
        self.sampler.device = device
        self.sampler.dtype = dtype
        self.sampler.model_id = DEFAULT_MODEL_ID
        self.sampler.lora_path = lora_path

        torch.backends.cuda.matmul.allow_tf32 = True

        # Load pipeline WITHOUT deleting text encoders yet
        from fluxfm_sampler_reward import HF_CACHE
        from flux_two_timestep import FluxPipelineTwoTimestep, add_dual_time_embedder
        print(f"Loading FLUX FlowMap pipeline for ReNO (resolution={resolution})")
        self.sampler.pipeline = FluxPipelineTwoTimestep.from_pretrained(
            DEFAULT_MODEL_ID,
            torch_dtype=dtype,
            cache_dir=HF_CACHE,
        ).to(device)

        # Add dual time embedder and load LoRA
        self.sampler.pipeline.transformer = add_dual_time_embedder(self.sampler.pipeline.transformer)
        self.sampler.pipeline.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")

        # Freeze all parameters
        for param in self.sampler.pipeline.transformer.parameters():
            param.requires_grad = False
        for param in self.sampler.pipeline.vae.parameters():
            param.requires_grad = False

        # Enable gradient checkpointing for memory efficiency
        self.sampler.pipeline.transformer.enable_gradient_checkpointing()

        self.sampler.vae = self.sampler.pipeline.vae
        self.sampler.vae_scale_factor = self.sampler.pipeline.vae_scale_factor

        # Prompt embedding cache
        self._prompt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._text_encoders_deleted = False

        # Precompute null embeddings
        with torch.no_grad():
            null_pe, null_ppe = self.sampler._encode_prompt_internal("")
            self._prompt_cache[""] = (null_pe, null_ppe)

        # Compute latent dimensions for this resolution
        # Match FluxPipeline.prepare_latents: height = 2 * (resolution // (vae_scale_factor * 2))
        H_latent = 2 * (self.resolution // (self.sampler.vae_scale_factor * 2))
        W_latent = H_latent  # square
        self._packed_seq_len = (H_latent // 2) * (W_latent // 2)
        self._latent_channels = 16  # FLUX uses 16 channels
        self._H_latent = H_latent
        self._W_latent = W_latent

        print(f"FlowMap pipeline loaded. Resolution={resolution}, "
              f"packed_seq_len={self._packed_seq_len}, "
              f"latent_shape=({self._latent_channels}, {H_latent}, {W_latent})")

    @property
    def latent_shape(self):
        """Shape for creating latents in packed form (what ReNO creates)."""
        return (1, self._packed_seq_len, self._latent_channels * 4)

    def precompute_embeddings(self, prompts: List[str]):
        """
        Encode all prompts, cache in dict, then delete text encoders to free VRAM.

        Must be called before reward models are loaded.
        """
        if self._text_encoders_deleted:
            raise RuntimeError("Text encoders already deleted. Cannot encode new prompts.")

        # Deduplicate
        unique_prompts = set(prompts) - set(self._prompt_cache.keys())
        print(f"Pre-encoding {len(unique_prompts)} unique prompts "
              f"({len(prompts)} total, {len(self._prompt_cache)} already cached)...")

        with torch.no_grad():
            for i, prompt in enumerate(unique_prompts):
                pe, ppe = self.sampler._encode_prompt_internal(prompt)
                self._prompt_cache[prompt] = (pe, ppe)
                if (i + 1) % 50 == 0:
                    print(f"  Encoded {i + 1}/{len(unique_prompts)} prompts...")

        print(f"Encoded all {len(self._prompt_cache)} unique prompts. Deleting text encoders...")

        # Delete text encoders
        del self.sampler.pipeline.text_encoder
        del self.sampler.pipeline.text_encoder_2
        del self.sampler.pipeline.tokenizer
        del self.sampler.pipeline.tokenizer_2
        self.sampler.pipeline.text_encoder = None
        self.sampler.pipeline.text_encoder_2 = None
        self.sampler.pipeline.tokenizer = None
        self.sampler.pipeline.tokenizer_2 = None
        gc.collect()
        torch.cuda.empty_cache()
        self._text_encoders_deleted = True
        print("Text encoders deleted.")

    def apply(self, latents, prompt, generator=None, num_inference_steps=1, **kwargs):
        """
        Generate image from latent noise (differentiable).

        ReNO calls this with packed latents of shape (1, seq_len, hidden_dim).
        We unpack → run FlowMap one-step → decode → return [0, 1] image.

        Args:
            latents: Packed latent tensor [B, seq_len, C*4] (from ReNO)
            prompt: Text prompt string
            generator: torch.Generator (unused, seed handled by ReNO)
            num_inference_steps: Number of steps (1 for FlowMap one-step)

        Returns:
            Image tensor [B, 3, H, W] in [0, 1] range
        """
        # Look up cached embeddings
        if prompt not in self._prompt_cache:
            if self._text_encoders_deleted:
                raise RuntimeError(f"Prompt '{prompt[:50]}...' not in cache and text encoders are deleted. "
                                   "Call precompute_embeddings() with all prompts first.")
            # Encode on the fly if text encoders still available
            with torch.no_grad():
                pe, ppe = self.sampler._encode_prompt_internal(prompt)
                self._prompt_cache[prompt] = (pe, ppe)

        prompt_embeds, pooled_prompt_embeds = self._prompt_cache[prompt]
        batch_size = latents.shape[0]

        # Expand embeddings to batch size
        if prompt_embeds.shape[0] != batch_size:
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.expand(batch_size, -1)

        # Unpack latents: [B, seq_len, C*4] → [B, C, H_latent, W_latent]
        z = self.sampler.pipeline._unpack_latents(
            latents, self.resolution, self.resolution, self.sampler.vae_scale_factor
        )

        # FlowMap one-step: predict velocity from t=1 to t=0
        # Set cached embeddings temporarily for predict_vector
        orig_pe = getattr(self.sampler, 'cached_prompt_embeds', None)
        orig_ppe = getattr(self.sampler, 'cached_pooled_prompt_embeds', None)
        self.sampler.cached_prompt_embeds = prompt_embeds
        self.sampler.cached_pooled_prompt_embeds = pooled_prompt_embeds

        u = self.sampler.predict_vector(
            z, t_cur=1.0,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=self.guidance_scale,
            t_next=0.0,  # FlowMap: predict mean velocity to t=0
        )

        # Restore original embeddings
        self.sampler.cached_prompt_embeds = orig_pe
        self.sampler.cached_pooled_prompt_embeds = orig_ppe

        # z0 = z - 1.0 * u (one step from t=1 to t=0)
        z0 = z - 1.0 * u

        # Decode to image (output is [-1, 1])
        image = self.sampler.decode(z0)

        # Convert [-1, 1] → [0, 1] for ReNO reward models
        image = (image + 1.0) / 2.0
        image = image.clamp(0, 1)

        # Cast to float16 to match reward model weights (FLUX outputs bfloat16)
        image = image.to(torch.float16)

        return image

    def multi_apply(self, latents, prompt, num_inference_steps=8):
        """
        Multi-step flow map generation from optimized latents (no grad).

        Like FLUX-schnell in ReNO: optimize with 1-step, generate final with multi-step.
        Just calls the pipeline's __call__ directly, which handles everything.

        Args:
            latents: Packed latent tensor [B, seq_len, C*4]
            prompt: Text prompt string
            num_inference_steps: Number of flow map steps
        """
        prompt_embeds, pooled_prompt_embeds = self._prompt_cache[prompt]

        # Call pipeline directly — it handles packing, timesteps, decoding
        output = self.sampler.pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.resolution,
            width=self.resolution,
            output_type="pt",
            return_dict=False,
        )
        # output is a tuple, first element is images [B, 3, H, W] in [0, 1]
        image = output[0]
        # Cast to float16 to match reward model weights (FLUX outputs bfloat16)
        image = image.to(torch.float16)
        return image
