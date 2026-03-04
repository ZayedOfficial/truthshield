---
title: TruthShield
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
python_version: "3.10"
pinned: false
license: apache-2.0
---

# 🛡️ TruthShield — The Anonymous Honesty Layer for Medicine

> **MedGemma Impact Challenge 2026 — Final Enterprise Release**

### 🔗 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/ZayedRehman/truthshield)

**80.9% of patients withhold medically relevant information** due to shame, fear of judgment, or clinical anxiety. TruthShield bridges this "Honesty Gap" using **MedGemma-4B Edge AI** to detect masked internal truths and provide clinicians with actionable, non-confrontational MI (Motivational Interviewing) openers.

---

## 🚀 Key Features (v2.0)
- **10-Point "Crystal Clear" Survey**: AI-personalized 10-MCQ diagnostic layer for deep truth discovery.
- **Absolute Zero Latency**: Simulation engine decoupled for instant (<50ms) presentation results.
- **100% Offline / Edge AI**: 4B MedGemma parameters running locally for absolute patient privacy.
- **SNOMED-CT & HL7 FHIR ready**: Structured discrepancy reports for seamless EHR integration.
- **High-Emotion Scenario Library**: 12+ pre-mapped cases (Cyberbullying, Elder Fraud, Veteran Trauma, etc.).

---

## ⚡ Quick Start (Perfect Run Presentation Mode)

```bash
# 1. Clone the Professional Repository
git clone https://github.com/ZayedOfficial/truthshield.git
cd truthshield

# 2. Install High-Performant Dependencies
pip install -r requirements.txt

# 3. Launch the Intelligence Portal (Simulation Mode)
python main.py

# 4. Open http://localhost:7860
#    → Select 'Cyberbullying' or 'Financial Fraud' 
#    → Experience the sub-50ms diagnostic flow.
```

---

## 📁 Project Architecture

```
truthshield/
├── app.py               # Main Entry point for Hugging Face
├── main.py              # Core Engine: Gradio UI & Discrepancy Logic
├── prompts.py           # MedGemma clinical prompt engineering & SIMULATED_ALERTS
├── scenarios.py         # 12+ High-fidelity clinical demo scenarios
├── integration.py       # HL7 FHIR & API Integration logic
├── questions.py         # Standard clinical question bank
├── requirements.txt     # Production dependencies
├── ANDROID_BUILD.md     # Mobile deployment guide (MLC-LLM)
├── VIDEO_SCRIPT.md      # Official 2.5-minute demo script
└── KAGGLE_WRITEUP.md    # Final competition documentation
```

---

## 🎯 Demo Clinical Cases

| Scenario | Severity | Masked Truth | Suggested Approach |
|----------|:--------:|--------------|-------------------|
| **Cyberbullying** | 🚨 HIGH | Adolescent trauma & somatic pain | "Many people your age deal with mean stuff online..." |
| **Financial Fraud** | 🔴 CRITICAL | Elder exploitation preventing med adherence | "Sometimes unexpected financial stress makes it hard..." |
| **Veteran Trauma** | 🟡 HIGH | Masked PTSD & hyper-vigilance | "Standard screens don't always capture the reality..." |
| **Hidden Grief** | 🔴 CRITICAL | Partner loss driving med discontinuation | "How have you been feeling since your partner passed?" |

---

## ⚖️ Disclaimer
TruthShield is a research entry for the MedGemma Impact Challenge. It serves as a clinical aid and does not replace professional medical judgment. 

**Privacy is the pill. Honesty is the cure.**
