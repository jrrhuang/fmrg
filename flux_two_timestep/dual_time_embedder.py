import torch
from diffusers import FluxPipeline
import copy

class DualTimeEmbedder(torch.nn.Module):
    def __init__(self, original_embedder):
        super().__init__()
        self.original_embedder = original_embedder
        self.second_embedder = copy.deepcopy(original_embedder)

    def forward(self, timestep, guidance, pooled_projections):
        # Check if timestep is a tuple of two inputs
        if timestep.shape[-1] == 2 and len(timestep.shape) >= 2:
            t1, t2 = timestep.unbind(dim=-1)
            # Embed both timesteps
            emb1 = self.original_embedder(t1, guidance, pooled_projections)
            emb2 = self.second_embedder(t2, guidance, pooled_projections)
            # Combine them (e.g., average)
            return (emb1 + emb2) / 2
        # raise ValueError(f"Timestep must be stacked timesteps, got {timestep.shape}")
        # Fallback for standard single timestep
        return self.original_embedder(timestep, guidance, pooled_projections)

def add_dual_time_embedder(single_time_flux_transformer):
    single_time_flux_transformer.time_text_embed = DualTimeEmbedder(single_time_flux_transformer.time_text_embed)
    print("Added dual time embeddings")
    return single_time_flux_transformer
