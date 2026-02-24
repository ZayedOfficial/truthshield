# 📱 TruthShield — Android Build Guide (MLC-LLM)

This guide covers compiling MedGemma-4B (AWQ 4-bit) for Android using MLC-LLM, producing a standalone APK that runs inference 100% on-device.

---

## Prerequisites

### Software
- **Python 3.10+**
- **Rust** (latest stable): https://rustup.rs/
- **Android Studio** (latest): https://developer.android.com/studio
  - Android SDK (API 28+)
  - Android NDK (r26+ recommended)
  - CMake 3.22+
- **Git LFS**: `git lfs install`

### Hardware
- Build machine: 16GB+ RAM, ~20GB free disk space
- Target device: Android phone with 6GB+ RAM (Snapdragon 8 Gen 1+ or equivalent)

---

## Step 1: Install MLC-LLM

```bash
# Install MLC-LLM from source (recommended for Android builds)
git clone https://github.com/mlc-ai/mlc-llm.git --recursive
cd mlc-llm

# Create a conda environment (recommended)
conda create -n mlc-build python=3.11
conda activate mlc-build

# Install Python package
pip install .

# Verify installation
python -c "import mlc_llm; print('MLC-LLM installed successfully')"
```

---

## Step 2: Prepare the MedGemma Model

### Option A: Use Pre-Quantized AWQ Model (Recommended)

```bash
# If you already ran setup_model.py, use the AWQ output
# The model should be at ./models/medgemma-4b-awq/

# Convert to MLC format
mlc_llm convert_weight \
  ./models/medgemma-4b-awq/ \
  --quantization q4f16_1 \
  --output ./models/medgemma-4b-mlc/
```

### Option B: Quantize from Scratch with MLC-LLM

```bash
# Download and convert in one step
mlc_llm convert_weight \
  google/medgemma-4b-it \
  --quantization q4f16_1 \
  --output ./models/medgemma-4b-mlc/
```

---

## Step 3: Generate MLC Configuration

```bash
# Generate the mlc-chat-config.json
mlc_llm gen_config \
  ./models/medgemma-4b-mlc/ \
  --quantization q4f16_1 \
  --conv-template gemma \
  --max-batch-size 1 \
  --output ./models/medgemma-4b-mlc/

# Compile the model library for Android
mlc_llm compile \
  ./models/medgemma-4b-mlc/ \
  --device android \
  --output ./dist/libs/medgemma-4b.tar
```

---

## Step 4: Set Up the Android Project

```bash
# Navigate to the MLC-LLM Android example app
cd mlc-llm/android/MLCChat

# Set environment variables
export ANDROID_HOME=$HOME/Android/Sdk  # Adjust for your system
export ANDROID_NDK=$ANDROID_HOME/ndk/26.2.11394342  # Use your NDK version

# Copy model weights and library
mkdir -p app/src/main/assets/models/medgemma-4b/
cp -r ../../models/medgemma-4b-mlc/* app/src/main/assets/models/medgemma-4b/
cp ../../dist/libs/medgemma-4b.tar app/src/main/assets/
```

### Configure the App for TruthShield

Edit `app/src/main/assets/app-config.json`:

```json
{
  "model_list": [
    {
      "model_url": "models/medgemma-4b/",
      "model_id": "medgemma-4b",
      "model_lib": "medgemma-4b.tar",
      "estimated_vram_bytes": 3000000000,
      "display_name": "TruthShield — MedGemma-4B"
    }
  ]
}
```

---

## Step 5: Build the APK

```bash
# Build debug APK
./gradlew assembleDebug

# The APK will be at:
# app/build/outputs/apk/debug/app-debug.apk

# Build release APK (for distribution)
./gradlew assembleRelease

# The release APK will be at:
# app/build/outputs/apk/release/app-release-unsigned.apk
```

---

## Step 6: Install on Device

```bash
# Via ADB (USB debugging must be enabled on the device)
adb install app/build/outputs/apk/debug/app-debug.apk

# Or transfer the APK file to the device and install manually
```

---

## Step 7: Test Offline

1. Open the TruthShield app on the Android device
2. **Turn off Wi-Fi and cellular data** (Airplane mode)
3. Type or paste a clinical scenario
4. Verify the model generates a response
5. Confirm response time is < 15 seconds on the device
6. ✅ Fully offline — no network traffic

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Out of memory` during build | Increase Gradle heap: `org.gradle.jvmargs=-Xmx4g` in `gradle.properties` |
| Model too large for APK | Use model streaming: place weights on device storage instead of bundling in APK |
| Slow inference on device | Ensure GPU delegate is enabled; some devices need Vulkan backend |
| NDK not found | Set `ANDROID_NDK` environment variable explicitly |
| `cmake not found` | Install CMake via Android Studio SDK Manager |

---

## Performance Benchmarks (Expected)

| Device | RAM | Inference Time | Notes |
|--------|:---:|:--------------:|-------|
| Pixel 8 Pro | 12GB | ~8-12s | Tensor G3, smooth |
| Samsung S24 | 12GB | ~7-10s | Snapdragon 8 Gen 3 |
| Pixel 7 | 8GB | ~12-18s | May need reduced context |
| Mid-range (6GB) | 6GB | ~15-25s | Functional but slower |

---

## Notes

- The MLC-LLM Android app provides a chat interface by default. For a custom TruthShield UI, you would extend the Android app with the survey + alert layout.
- Model weights are ~2.5GB after 4-bit quantization — fits on most modern Android devices.
- For the competition video, using the default MLC chat interface with TruthShield prompts is sufficient to demonstrate on-device inference.
