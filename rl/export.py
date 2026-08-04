"""Turn a training checkpoint into the two files the website loads.

Writes into public/policy/:
    manifest.json   shapes + quantization scale
    weights.bin     raw little-endian weights, tables concatenated

The layout here and the reader in src/game/policy.ts are one contract.
The reader validates the total weight count against the manifest, so a
mismatch fails loudly at load rather than silently mis-indexing.

Usage:
    python export.py --checkpoint checkpoints/web-5tuple.npz
    python export.py --checkpoint checkpoints/web-5tuple.npz --dtype float32
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from engine import face_value
from ntuple import NTupleNetwork

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "public", "policy")


def quantize(weights: np.ndarray, dtype: str) -> tuple[np.ndarray, float]:
    """Compress weights to `dtype`, returning (stored, scale).

    A single global scale keeps the browser side to one multiply at the
    end of the sum, instead of per-table bookkeeping.
    """
    if dtype == "float32":
        return weights.astype("<f4"), 1.0

    peak = float(np.abs(weights).max())
    if peak == 0.0:
        raise SystemExit("refusing to export an all-zero network — did training run?")

    limit = 32767 if dtype == "int16" else 2147483647
    scale = peak / limit

    stored = np.rint(weights / scale)
    stored = np.clip(stored, -limit, limit)

    # int16 buys a 2x smaller download at the cost of resolution. Worth
    # checking the damage is negligible rather than assuming it.
    error = np.abs(stored * scale - weights).max()
    print(f"  quantization: scale {scale:.6g}, max error {error:.4g} (peak |w| {peak:.1f})")

    return stored.astype("<i2" if dtype == "int16" else "<i4"), scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--dtype", choices=["int16", "int32", "float32"], default="int16")
    parser.add_argument("--note", default="")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Recorded in the manifest for display.")
    parser.add_argument("--mean-score", type=float, default=None)
    parser.add_argument("--win-rate", type=float, default=None)
    args = parser.parse_args()

    net = NTupleNetwork.load(args.checkpoint)
    config = net.config
    print(config.describe())

    stored, scale = quantize(net.weights, args.dtype)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    weights_path = os.path.join(out_dir, "weights.bin")
    stored.tofile(weights_path)

    manifest = {
        "format": "ntuple-v1",
        "alphabet": config.alphabet,
        "tuples": [list(map(int, t)) for t in config.tuples],
        "symmetric": bool(config.symmetric),
        "quant": {"dtype": args.dtype, "scale": scale},
        "weights": "weights.bin",
        "meta": {
            k: v
            for k, v in {
                "trainedEpisodes": args.episodes,
                "meanScore": args.mean_score,
                "winRate": args.win_rate,
                "note": args.note or None,
            }.items()
            if v is not None
        },
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    size_mb = os.path.getsize(weights_path) / 1e6
    print(f"\nwrote {manifest_path}")
    print(f"wrote {weights_path}  ({size_mb:.1f} MB)")

    if size_mb > 8:
        print(
            "\n  warning: that's a heavy download for a personal site.\n"
            "  Consider --dtype int16, shorter tuples, or a smaller alphabet."
        )

    print("\nCommit public/policy/ and the page will pick it up automatically.")


if __name__ == "__main__":
    main()
