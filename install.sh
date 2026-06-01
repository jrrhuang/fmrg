#!/bin/bash
set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )"
ENV_FILE="$SCRIPT_DIR/environment.yml"
ENV_NAME="fmrg_env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "environment.yml not found at $ENV_FILE"
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    read -p "$ENV_NAME already exists. Remove and reinstall? (y/N) " -n 1 -r REPLY
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 0
    conda env remove -n "$ENV_NAME" -y
fi

conda env create -f "$ENV_FILE" -n "$ENV_NAME"
conda activate "$ENV_NAME"

# CLIP, image-reward, and hpsv2 need --no-build-isolation because their
# setup.py modules import pkg_resources, which the latest setuptools dropped.
pip install --no-build-isolation git+https://github.com/openai/CLIP.git
pip install --no-build-isolation image-reward hpsv2

# hpsv2's PyPI wheel omits its vendored open_clip BPE vocab file. Copy it from
# the openai-clip install.
HPSV2_OC="$(python -c "import hpsv2,os;print(os.path.join(os.path.dirname(hpsv2.__file__),'src','open_clip'))")"
CLIP_BPE="$(python -c "import clip,os;print(os.path.join(os.path.dirname(clip.__file__),'bpe_simple_vocab_16e6.txt.gz'))")"
if [ -f "$CLIP_BPE" ] && [ -d "$HPSV2_OC" ] && [ ! -f "$HPSV2_OC/bpe_simple_vocab_16e6.txt.gz" ]; then
    cp "$CLIP_BPE" "$HPSV2_OC/"
fi

python -c "import torch, diffusers, transformers; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} | diffusers {diffusers.__version__} | transformers {transformers.__version__}')"
