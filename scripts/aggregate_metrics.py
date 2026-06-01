"""
Dataset-level aggregator for inverse-problem reconstructions.

Reads per-image metrics from <save_dir>/<task_op>/progress/<id>/metadata.json,
computes dataset-level FID + KID against the ground-truth directory, and writes
the summary to <save_dir>/<task_op>/aggregate_metrics.json.

Usage:
    python scripts/aggregate_metrics.py --save_dir <run_dir> --gt_path <gt_dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np


def find_task_op_dir(save_dir: Path) -> Path:
    cands = [d for d in save_dir.iterdir() if d.is_dir() and (d / "recon").exists()]
    if len(cands) != 1:
        raise ValueError(f"expected exactly one task_op subdir under {save_dir}, got {[c.name for c in cands]}")
    return cands[0]


def aggregate_per_image(task_op_dir: Path) -> dict:
    progress_dir = task_op_dir / "progress"
    if not progress_dir.exists():
        return {"per_image_error": f"missing {progress_dir}"}
    metric_files = sorted(progress_dir.glob("*/metadata.json"))
    rows = []
    for mf in metric_files:
        try:
            data = json.loads(mf.read_text())
            m = data.get("metrics")
            if m:
                rows.append(m)
        except Exception:
            pass
    out = {"num_images": len(rows)}
    for k in ("psnr", "ssim", "lpips"):
        vals = [r[k] for r in rows if k in r and r[k] is not None]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
    return out


def compute_fid_kid(recon_dir: Path, gt_dir: Path, device: str = "cuda") -> dict:
    import torch
    from PIL import Image
    import torchvision.transforms as T
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    recon_files = sorted(p for p in recon_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    gt_lookup = {p.stem: p for p in Path(gt_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    common = [(r, gt_lookup[r.stem]) for r in recon_files if r.stem in gt_lookup]
    if not common:
        return {"fid_kid_error": "no matching filenames between recon and gt directories"}
    if len(common) < 2:
        return {"fid_kid_skipped": f"only {len(common)} image pair(s) — FID/KID need ≥ 2", "num_pairs": len(common)}

    transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])

    def to_uint8_batch(paths):
        ts = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            ts.append((transform(img) * 255).to(torch.uint8))
        return torch.stack(ts)

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(subset_size=min(50, len(common)), normalize=False).to(device)

    BATCH = 32
    for s in range(0, len(common), BATCH):
        chunk = common[s:s + BATCH]
        r_b = to_uint8_batch([r for r, _ in chunk]).to(device)
        g_b = to_uint8_batch([g for _, g in chunk]).to(device)
        fid.update(r_b, real=False); fid.update(g_b, real=True)
        kid.update(r_b, real=False); kid.update(g_b, real=True)

    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    return {"fid": float(fid_score), "kid_mean": float(kid_mean.item()),
            "kid_std": float(kid_std.item()), "num_pairs": len(common)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save_dir", required=True, type=Path,
                    help="Output dir from solve_inverse_problem.py (contains <task_op>/{recon,progress,label}).")
    ap.add_argument("--gt_path", required=True, type=Path,
                    help="Directory of ground-truth images at the inference resolution (256 by default).")
    ap.add_argument("--output_json", default=None, type=Path,
                    help="Output path for the summary JSON (default: <save_dir>/<task_op>/aggregate_metrics.json).")
    ap.add_argument("--skip_fid_kid", action="store_true",
                    help="Skip dataset-level FID/KID (still reports per-image PSNR/SSIM/LPIPS).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    task_op = find_task_op_dir(args.save_dir)
    print(f"[task_op] {task_op}")
    per_image = aggregate_per_image(task_op)
    fid_kid = {} if args.skip_fid_kid else compute_fid_kid(task_op / "recon", args.gt_path, args.device)
    summary = {
        "save_dir": str(args.save_dir),
        "task_op": task_op.name,
        "gt_path": str(args.gt_path),
        **per_image,
        **fid_kid,
    }
    out_json = args.output_json or (task_op / "aggregate_metrics.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
