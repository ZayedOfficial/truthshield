# 🛡️ TruthShield — The Anonymous Honesty Layer for Medicine

> **MedGemma Impact Challenge 2026 Submission**

**60–80% of patients lie to their doctors** about alcohol, drugs, medication adherence, and suicidal thoughts (Levy et al., JAMA Network Open, 2018). These lies can be fatal.

TruthShield is a fully anonymous pre-visit survey system that uses **MedGemma-4B** to detect life-threatening discrepancies between what patients privately report and their clinical records — generating non-confrontational clinician alerts. **100% offline, on-device.**

---

## ⚡ Quick Start (Simulation Mode — No Model Needed)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/truthshield.git
cd truthshield

# 2. Install dependencies
pip install gradio transformers torch

# 3. Run (simulation mode — pre-generated alerts, no GPU needed)
python main.py

# 4. Open http://localhost:7860 in your browser
#    → Select a demo scenario → Load → Analyze
```

**That's it.** The app runs fully offline in simulation mode with pre-generated MedGemma alerts for 3 clinical scenarios.

---

## 🧠 Full Setup (Real MedGemma Inference)

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM (or Apple Silicon with 8GB+ RAM)
- [HuggingFace account](https://huggingface.co) with access to `google/medgemma-4b-it`

### Step-by-Step

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Download and quantize MedGemma-4B to AWQ 4-bit
python setup_model.py --hf-token YOUR_HUGGINGFACE_TOKEN

# 3. Run with real model inference
python main.py --model-path ./models/medgemma-4b-awq

# 4. Open http://localhost:7860
```

### Offline Verification

1. **Disconnect from the internet** (turn off Wi-Fi / unplug ethernet)
2. Run `python main.py --model-path ./models/medgemma-4b-awq`
3. Load a demo scenario and click "Analyze"
4. Confirm the clinician alert generates in < 10 seconds
5. ✅ Fully offline — no network calls made

---

## 📁 Project Structure

```
truthshield/
├── main.py              # Gradio app — core UI and inference
├── prompts.py           # MedGemma prompt templates
├── scenarios.py         # 3 pre-built clinical demo scenarios
├── setup_model.py       # Model download + AWQ quantization
├── requirements.txt     # Python dependencies
├── ANDROID_BUILD.md     # MLC-LLM Android APK build guide
├── VIDEO_SCRIPT.md      # 2:45 demo video script
├── KAGGLE_WRITEUP.md    # 3-page competition write-up
└── README.md            # This file
```

---

## 🎯 Demo Scenarios

| Scenario | Severity | What the Patient Hides |
|----------|:--------:|------------------------|
| 🚨 Teen Suicide Prevention | **CRITICAL** | 16yo reports active suicidal ideation on anonymous survey; denies everything in front of parent |
| 🍺 Hidden Alcohol Misuse | **HIGH** | 45yo diabetic reports 15+ drinks/week anonymously; tells doctor "2-3 socially" |
| 💊 Silent Med Discontinuation | **HIGH** | 62yo post-MI patient stopped all cardiac meds 3 months ago without telling cardiologist |

---

## 📱 Android Deployment

See [ANDROID_BUILD.md](ANDROID_BUILD.md) for step-by-step instructions to compile MedGemma via MLC-LLM and build a standalone Android APK.

---

## 📚 Key References

1. Levy AG, et al. **"Prevalence of and Factors Associated With Patient Nondisclosure of Medically Relevant Information to Clinicians."** JAMA Network Open. 2018;1(7):e185293.
2. Google Health AI. **MedGemma: Open models for medical text and image understanding.** 2025.
3. MLC-LLM Team. **Machine Learning Compilation for Large Language Models.** mlc.ai, 2024.

---

## ⚖️ Disclaimer

TruthShield is a research demonstration for the MedGemma Impact Challenge. It is **not intended for clinical use** without proper regulatory approval, validation, and IRB review. All clinical decisions remain with the treating provider.

---

## 📄 License

Apache 2.0
