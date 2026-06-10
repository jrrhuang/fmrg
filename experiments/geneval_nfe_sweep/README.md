# GenEval NFE sweep

## Run

```bash
bash generate_fmrg.sh
bash generate_bon.sh
bash reno/reproduce.sh

GENEVAL_REPO=$HOME/geneval GENEVAL_MODELS=/path/to/geneval_models \
  bash evaluate.sh ../../results/geneval_nfe_sweep

python show_results.py ../../results/geneval_nfe_sweep
```

## Reference values (4-seed mean)

| Method | NFE → score |
|---|---|
| FMRG-J | 6→0.695, 11→0.730, 21→0.770, 29→0.781, 41→0.786, 61→0.789, 81→0.797, 100→0.800 |
| FMRG-E | 5→0.682, 9→0.722, 18→0.729, 30→0.744, 59→0.760, 99→0.766 |
| Best-of-N (flow map) | 4→0.648, 8→0.692, 16→0.715, 32→0.739, 64→0.746, 128→0.758 |
| ReNO | 9→0.678, 18→0.699, 33→0.714, 58→0.718, 108→0.719 |
| FLUX.1-dev (50 NFE) | 0.662 |

## GenEval evaluator setup

```bash
git clone https://github.com/djghosh13/geneval.git $HOME/geneval
conda create -n geneval python=3.10 -y
conda activate geneval
pip install torch==2.1.2+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install openmim && mim install mmcv-full==1.7.2 && pip install mmdet==2.28.2
pip install -r $HOME/geneval/requirements.txt
bash $HOME/geneval/evaluation/download_models.sh /path/to/geneval_models
```

Activate the `geneval` env before running `evaluate.sh`.
