"""
TruthShield — Model Download & AWQ Quantization Script

Downloads MedGemma-4B-instruct from HuggingFace and quantizes to
AWQ 4-bit for fast edge inference.

Prerequisites:
    pip install transformers autoawq torch accelerate sentencepiece protobuf

Usage:
    python setup_model.py --hf-token YOUR_TOKEN
    python setup_model.py --hf-token YOUR_TOKEN --output-dir ./models/medgemma-4b-awq

You need a HuggingFace token with access to google/medgemma-4b-it.
Request access at: https://huggingface.co/google/medgemma-4b-it
"""

import argparse
import os
import sys
import time


def check_dependencies():
    """Verify all required packages are installed."""
    missing = []
    # AutoAWQ is optional for Safe Download mode
    for pkg in ["torch", "transformers", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"  Run: pip install {' '.join(missing)}")
        sys.exit(1)

    print("[OK] Core dependencies found.")


def download_and_quantize(hf_token: str, output_dir: str, model_id: str):
    """Download the model using snapshot_download for disk efficiency."""

    from huggingface_hub import snapshot_download
    print(f"\n{'='*60}")
    print(f"  TruthShield Model Setup")
    print(f"  Model: {model_id}")
    print(f"  Mode: Disk-Efficient Safe Download")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # ── Step 1: Download model directly ─────────────────────────────────
    print("[1/2] Downloading model weights directly to output folder...")
    print("      (Avoiding double-saving to save disk space)")
    start = time.time()

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            token=hf_token,
            local_dir_use_symlinks=False,  # Copy files directly
            ignore_patterns=["*.pth", "*.bin"] if ".safetensors" in model_id else [] # Prefer safetensors
        )
        elapsed = time.time() - start
        print(f"      Downloaded directly in {elapsed:.0f}s\n")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)

    # ── Step 2: Validate ────────────────────────────────────────────────
    print("[2/2] Validating model files...")
    if os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"      ✅ Model files synchronized successfully to: {output_dir}")
    else:
        print(f"      ❌ Missing config.json in {output_dir}")
        sys.exit(1)

    print(f"      TruthShield will use bitsandbytes 4-bit loading at runtime to save VRAM.")

    # ── Step 4: Final Message ───────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  ✅ SYNCHRONIZATION COMPLETE!")
    print(f"  Model weights saved to: {output_dir}")
    print(f"  ")
    print(f"  TruthShield will now automatically detect and load these weights.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and quantize MedGemma-4B for TruthShield"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        required=True,
        help="HuggingFace token with access to google/medgemma-4b-it",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/medgemma-4b-it",
        help="HuggingFace model ID (default: google/medgemma-4b-it)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/medgemma-4b-awq",
        help="Directory to save quantized model (default: ./models/medgemma-4b-awq)",
    )
    args = parser.parse_args()

    check_dependencies()
    download_and_quantize(args.hf_token, args.output_dir, args.model_id)


if __name__ == "__main__":
    main()
