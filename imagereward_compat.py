"""
Monkey-patch transformers 5.x to restore APIs removed since 4.x.
This allows ImageReward (which depends on transformers 4.x APIs) to work
without modifying any ImageReward source files.

Import this module BEFORE importing ImageReward:
    import imagereward_compat
    import ImageReward
"""

import torch
from typing import List, Set, Tuple, Optional, Union

# ============================================================================
# 1. Restore apply_chunking_to_forward and prune_linear_layer in modeling_utils
#    (moved to pytorch_utils in transformers 5.x)
# ============================================================================
import transformers.modeling_utils as mu
import transformers.pytorch_utils as pu

if not hasattr(mu, 'apply_chunking_to_forward'):
    mu.apply_chunking_to_forward = pu.apply_chunking_to_forward

if not hasattr(mu, 'prune_linear_layer'):
    mu.prune_linear_layer = pu.prune_linear_layer

# ============================================================================
# 2. Restore find_pruneable_heads_and_indices (removed entirely in 5.x)
# ============================================================================
if not hasattr(mu, 'find_pruneable_heads_and_indices'):
    if hasattr(pu, 'find_pruneable_heads_and_indices'):
        # transformers 5.x kept it in pytorch_utils but removed from modeling_utils
        mu.find_pruneable_heads_and_indices = pu.find_pruneable_heads_and_indices
    else:
        # newer transformers removed it from both — re-implement the 4.x version
        def find_pruneable_heads_and_indices(
            heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
        ) -> Tuple[Set[int], torch.LongTensor]:
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index: torch.LongTensor = torch.arange(len(mask))[mask].long()
            return heads, index

        mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

# ============================================================================
# 3. Restore get_head_mask and _convert_head_mask_to_5d on PreTrainedModel
#    (removed in transformers 5.x)
# ============================================================================
if not hasattr(mu.PreTrainedModel, 'get_head_mask'):
    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = _convert_head_mask_to_5d(self, head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    mu.PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d
    mu.PreTrainedModel.get_head_mask = get_head_mask

# ============================================================================
# 4. Restore additional_special_tokens / additional_special_tokens_ids
#    on tokenizers (__getattr__ intercepts lookup in transformers 5.x)
# ============================================================================
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# transformers 5.x intercepts attribute lookup via PreTrainedTokenizerBase.__getattr__;
# transformers 4.x has no such class attribute (ImageReward works natively there),
# so only install this shim when __getattr__ actually exists.
_original_tokenizer_getattr = getattr(PreTrainedTokenizerBase, "__getattr__", None)

def _patched_tokenizer_getattr(self, key):
    if key == 'additional_special_tokens_ids':
        tokens = getattr(self, '_additional_special_tokens',
                         getattr(self, 'additional_special_tokens_list', []))
        if not tokens:
            tokens = [t for t in getattr(self, 'added_tokens_encoder', {})
                      if t not in ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]')]
        return self.convert_tokens_to_ids(tokens)
    if key == 'additional_special_tokens':
        tokens = getattr(self, '_additional_special_tokens',
                         getattr(self, 'additional_special_tokens_list', []))
        if not tokens:
            tokens = [t for t in getattr(self, 'added_tokens_encoder', {})
                      if t not in ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]')]
        return tokens
    return _original_tokenizer_getattr(self, key)

if _original_tokenizer_getattr is not None:
    PreTrainedTokenizerBase.__getattr__ = _patched_tokenizer_getattr

# ============================================================================
# 5. Ensure _tied_weights_keys and all_tied_weights_keys default to {}
#    (transformers 5.x tie_weights() calls .items() on them, crashes if None)
# ============================================================================
_original_tie_weights = mu.PreTrainedModel.tie_weights

def _patched_tie_weights(self, **kwargs):
    if getattr(self, '_tied_weights_keys', None) is None:
        self._tied_weights_keys = {}
    if not hasattr(self, 'all_tied_weights_keys') or getattr(self, 'all_tied_weights_keys', None) is None:
        self.all_tied_weights_keys = {}
    return _original_tie_weights(self, **kwargs)

mu.PreTrainedModel.tie_weights = _patched_tie_weights

_original_post_init = mu.PreTrainedModel.post_init

def _patched_post_init(self):
    if not hasattr(self, 'all_tied_weights_keys'):
        self.all_tied_weights_keys = {}
    if getattr(self, '_tied_weights_keys', None) is None:
        self._tied_weights_keys = {}
    return _original_post_init(self)

mu.PreTrainedModel.post_init = _patched_post_init

# ============================================================================
# 6. Fix CLIPProcessor.__call__ for text-only usage
#    In transformers 5.x, CLIPProcessor(prompt) interprets str as image URL.
#    Patch to route string args to text processing when no images kwarg given.
# ============================================================================
try:
    from transformers import CLIPProcessor
    _original_clip_processor_call = CLIPProcessor.__call__

    def _patched_clip_processor_call(self, *args, **kwargs):
        # If first positional arg is a string and 'images' not in kwargs,
        # treat it as text input (transformers 4.x behavior)
        if args and isinstance(args[0], str) and 'images' not in kwargs and 'text' not in kwargs:
            kwargs['text'] = args[0]
            args = args[1:]
        return _original_clip_processor_call(self, *args, **kwargs)

    CLIPProcessor.__call__ = _patched_clip_processor_call
except ImportError:
    pass

# ============================================================================
# 7. Fix CLIPModel.get_image_features / get_text_features return type
#    In transformers 5.x these return BaseModelOutputWithPooling instead of tensor
# ============================================================================
try:
    from transformers import CLIPModel
    _original_get_image_features = CLIPModel.get_image_features
    _original_get_text_features = CLIPModel.get_text_features

    def _patched_get_image_features(self, *args, **kwargs):
        result = _original_get_image_features(self, *args, **kwargs)
        if hasattr(result, 'image_embeds'):
            return result.image_embeds
        if hasattr(result, 'pooler_output'):
            return result.pooler_output
        return result

    def _patched_get_text_features(self, *args, **kwargs):
        result = _original_get_text_features(self, *args, **kwargs)
        if hasattr(result, 'text_embeds'):
            return result.text_embeds
        if hasattr(result, 'pooler_output'):
            return result.pooler_output
        return result

    CLIPModel.get_image_features = _patched_get_image_features
    CLIPModel.get_text_features = _patched_get_text_features
except ImportError:
    pass
