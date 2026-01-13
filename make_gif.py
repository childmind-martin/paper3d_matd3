#!/usr/bin/env python3
"""
Simple helper to stitch a sequence of images into a GIF.

Usage:
  python make_gif.py --pattern "logs/run_*/frames/*.png" --output demo.gif --fps 12 --limit 300
"""

import argparse
import sys
from pathlib import Path

try:
    import imageio.v2 as imageio
except Exception as e:  # pragma: no cover - import guard
    print("imageio is required (pip install imageio). Error:", e)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Create a GIF from a set of images.")
    parser.add_argument(
        "--pattern",
        required=True,
        help="Glob pattern for input images (e.g., 'frames/*.png' or 'logs/run/frames/*.jpg').",
    )
    parser.add_argument(
        "--output",
        default="out.gif",
        help="Output GIF path (default: out.gif).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for the GIF (default: 10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of frames (after sorting).",
    )
    args = parser.parse_args()

    paths = sorted(Path().glob(args.pattern))
    if not paths:
        print(f"No images matched pattern: {args.pattern}")
        sys.exit(1)

    if args.limit is not None and args.limit > 0:
        paths = paths[: args.limit]

    frames = []
    for p in paths:
        try:
            frames.append(imageio.imread(p))
        except Exception as e:
            print(f"Skip {p}: {e}")

    if not frames:
        print("No valid frames read; abort.")
        sys.exit(1)

    duration = 1.0 / max(args.fps, 1e-3)
    imageio.mimsave(args.output, frames, duration=duration)
    print(f"Wrote GIF: {args.output} ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
