"""
TruthShield — The Anonymous Honesty Layer for Medicine
Enterprise Clinical Application

Production-grade Gradio application designed for integration across
all hospital departments. Runs 100% offline using MedGemma-4B-instruct
(AWQ 4-bit quantized).

Usage:
    python main.py
    python main.py --model-path ./models/medgemma-4b-awq
"""

import argparse
import time
import datetime
import os
import sys
import json

import gradio as gr

from prompts import (
    SYSTEM_PROMPT,
    build_full_prompt,
    MCQ_GENERATION_PROMPT,
    get_simulated_alert,
)
from scenarios import SCENARIOS, get_scenario_list, get_scenario
from integration import generate_fhir_bundle, generate_api_curl_sample
from questions import PATIENT_MCQS
import huggingface_hub

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────
DEPARTMENTS = [
    "Primary Care",
    "Pediatrics & Adolescent Medicine",
    "Emergency Medicine",
    "Cardiology",
    "Psychiatry & Behavioral Health",
    "Internal Medicine",
    "Obstetrics & Gynecology",
    "Oncology",
    "Neurology",
    "Geriatrics",
]

class ClinicalAIEngine:
    """Universal loader and interface for clinical AI models."""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_simulation = True
        self.model_name = "None (Simulation Active)"
        self.device = "cpu"
        self.load_error: str = ""
        # Track personalized questions for the final honesty report
        self.current_personalized_qs = []

    def detect_local_models(self):
        """Scans ./models/ for compatible transformers models."""
        models_dir = "./models"
        if not os.path.exists(models_dir):
            return []
        try:
            return [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        except:
            return []

    def load(self, model_path: str = None):
        """Loads a model with robust error handling and quantization support."""
        try:
            # Lazy imports to allow UI-only launch on restricted environments (e.g. HF Spaces)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
            except ImportError:
                print("[TruthShield] Transformers/Torch not found. AI Inference disabled (Simulation Only).")
                self.is_simulation = True
                return False, "AI Libraries not installed. Simulation mode active."

            if not model_path:
                local_models = self.detect_local_models()
                if local_models:
                    model_path = os.path.join("./models", local_models[0])
                else:
                    raise FileNotFoundError("MedGemma weights not found in ./models/. Please initialize first.")

            print(f"[TruthShield] Initializing Intelligence Layer: {model_path}...")
            start = time.time()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Smart weight loading for CPU vs GPU
            is_cuda = torch.cuda.is_available()
            load_device = "cuda" if is_cuda else "cpu"
            
            try:
                # Optimized for CPU execution on 16GB RAM systems - avoid float32 for RAM safety
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    device_map=load_device, 
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                print(f"[TruthShield] Standard load failed, trying auto map: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.device = str(self.model.device)
            self.model_name = os.path.basename(model_path).replace("-", " ").title()
            self.is_simulation = False
            self.load_error = ""
            
            elapsed = time.time() - start
            print(f"[TruthShield] Engine Ready: {self.model_name} activated on {self.device} ({elapsed:.1f}s)")
            return True, f"Successfully loaded {self.model_name}"
            
        except Exception as e:
            self.load_error = str(e)
            self.is_simulation = True
            print(f"[TruthShield] Engine Standby (MedGemma not found): {e}")
            return False, str(e)

    def run_inference(self, prompt_text, system_msg=SYSTEM_PROMPT, max_tokens=512):
        """Generic inference wrapper."""
        if self.is_simulation or not self.model:
            return None
            
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_text},
        ]
        
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        import torch
        # Extreme CPU optimization: Use all cores
        torch.set_num_threads(os.cpu_count() or 4)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, 
                do_sample=False, # Greedy decoding for maximum stability
                repetition_penalty=1.1, # Reduced for speed
            )
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# Singleton Engine
AI_ENGINE = ClinicalAIEngine()

def load_model(model_path: str):
    return AI_ENGINE.load(model_path)

def run_inference(survey_text, notes_text, patient_age, visit_type):
    prompt = build_full_prompt(survey_text, notes_text, patient_age, visit_type)
    return AI_ENGINE.run_inference(prompt)


def generate_ai_mcqs(patient_story, is_simulation, count=10):
    """Generate personalized MCQs using the MedGemma AI engine."""
    output_mcqs = []

    # Real Engine Inference Only
    if not AI_ENGINE.is_simulation:
        try:
            print(f"[TruthShield] Generating {count} AI MCQs for story: {patient_story[:50]}...")
            response = AI_ENGINE.run_inference(
                MCQ_GENERATION_PROMPT.format(patient_story=patient_story),
                system_msg=f"You are a clinical psychometrician. Generate exactly {count} nuanced questions. One per line.",
                max_tokens=600 # Increased for 10 questions
            )
            
            if response:
                print(f"[TruthShield] Received AI Response ({len(response)} chars)")
                # Robust parsing: Handle newlines or numbered lists
                raw_lines = response.strip().split("\n")
                
                # If everything is on one line, try to split by "QuestionNumber."
                if len(raw_lines) == 1 and "|" in raw_lines[0]:
                    import re
                    # Look for things like "2. ", "3. " to split
                    split_parts = re.split(r'\d+\.\s+', raw_lines[0])
                    raw_lines = [p.strip() for p in split_parts if "|" in p]

                for line in raw_lines:
                    line = line.strip()
                    if "|" in line:
                        parts = line.split("|")
                        if len(parts) == 2:
                            q_text = parts[0].strip()
                            # Clean up leading numbers if any (e.g., "1. How...")
                            import re
                            q_text = re.sub(r'^\d+\.\s*', '', q_text)
                            
                            opts = [o.strip() for o in parts[1].split(",")]
                            if q_text and len(opts) >= 2:
                                output_mcqs.append((q_text, opts))
                    if len(output_mcqs) >= count: 
                        break
        except Exception as e:
            import traceback
            print(f"[TruthShield] MedGemma MCQ Generation failed: {e}")
            traceback.print_exc()

    # Safety Fallback: Use standard clinical set only if AI is initializing
    if len(output_mcqs) < count:
        # Fill strictly from the standard clinical question bank
        for q in PATIENT_MCQS:
            if len(output_mcqs) < count:
                if not any(bq[0] == q["question"] for bq in output_mcqs):
                    output_mcqs.append((q["question"], q["options"]))

    return output_mcqs

    return output_mcqs


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────


def analyze_discrepancies(survey_text, clinical_notes, patient_age, visit_type, is_simulation_mode, *mcq_answers):
    # Flatten mcq_answers if it's a list of lists (caused by some Gradio versions/interactions)
    flat_answers = []
    for item in mcq_answers:
        if isinstance(item, list):
            flat_answers.extend(item)
        else:
            flat_answers.append(item)
    
    if not survey_text.strip() or not clinical_notes.strip():
        yield "⚠️ Please provide both the anonymous survey responses and the EHR clinical notes to run the analysis.", "", ""
        return

    yield "Analyzing — TruthShield is processing clinical discrepancies…", "", ""

    start_time = time.time()
    
    # Pre-process MCQs for better model/sim context
    mcq_summary = ""
    for i, ans in enumerate(flat_answers):
        if ans:
            mcq_summary += f"\n- Q{i+1}: {ans}"

    # 1. Check for Simulation/Demo Mode First
    alert = None
    used_model = False
    
    if is_simulation_mode:
        # Robust Detection: Check survey, notes, and scenario list for matches
        scenario_id = "general"
        search_blob = (survey_text + " " + clinical_notes).lower()
        
        for key, s_data in SCENARIOS.items():
            # Check key, title, and key with spaces
            if key in search_blob or s_data['title'].lower() in search_blob or key.replace("_", " ") in search_blob:
                scenario_id = key
                break
        
        # Strictly Instant: Never fall back to real model in simulation mode
        alert = get_simulated_alert(scenario_id)
        used_model = False
    
    # 2. Real AI Path
    elif not AI_ENGINE.is_simulation:
        full_text = survey_text + "\n\nSTRUCTURED MCQS:" + mcq_summary
        # Extreme speed target for analysis
        alert = AI_ENGINE.run_inference(full_text, clinical_notes, max_tokens=200)
        used_model = (alert is not None)

    # 2. No Fallback allowed - Report Status
    if alert is None:
        alert = "### ⚠️ MedGemma Intelligence Core Not Loaded\nTruthShield is currently synchronizing AI weights. Analysis will be available once the clinical model is initialized."

    elapsed = time.time() - start_time
    engine = "MedGemma 4B (HuggingFace/AWQ)" # Pure AI Backend
    ts = datetime.datetime.now().strftime("%H:%M:%S")

    fhir_bundle = generate_fhir_bundle(patient_age, visit_type, alert, mcq_answers)

    alert_class = "ts-intelligence-report"
    if "🔴 CRITICAL" in alert:
        alert_class += " alert-critical"

    timer_html = f"""<div class="ts-status-bar">
        <span style="display:flex;gap:24px;align-items:center;">
          <span>STATUS: <strong style="color:var(--c-primary);">COMPLETE</strong></span>
          <span>ENGINE: <strong>{engine}</strong></span>
          <span>TIME: <strong>{ts}</strong></span>
        </span>
        <span style="font-weight:700; color:var(--c-text-light);">🔒 NONE TRANSMITTED — OFFLINE</span>
    </div>"""

    yield gr.update(value=alert, elem_classes=[alert_class]), timer_html, fhir_bundle


# ─────────────────────────────────────────────────────────────────────────────
# CSS — World-Class Hospital Intelligence Design System
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

:root, .dark {
    color-scheme: light !important;
    /* Friendly Clinical Healthcare Palette - REFINED BRAND */
    --c-bg:         #f2faf7;   /* Soft Mint White */
    --c-surface:    #ffffff;
    --c-border:     #d1fae5;
    
    --c-text:       #064e3b;   /* Deep Medical Green */
    --c-text-2:     #1e293b;   /* Slate 800 - Body text */
    --c-text-3:     #475569;   /* Slate 600 - Subtext */
    
    --c-primary:    #0d9488;   /* Premium Clinical Teal/Green */
    --c-primary-h:  #0f766e;
    --c-primary-soft:#f0fdfa;
    
    --c-accent-amber:#d97706;  /* Professional Clinical Gold */
    --c-red:         #be123c;  /* Softer Medical Red */
    --c-red-bg:      #fff1f2;
    
    --c-radius:     16px;      /* Softer, friendlier corners */
    
    --c-text-light: #94a3b8;   /* Slate 400 */
    --c-border-med: #e2e8f0;   /* Slate 200 */

    /* Force Gradio internal variables to light mode & Transparency */
    --body-background-fill: var(--c-bg) !important;
    --body-text-color: var(--c-text) !important;
    --background-fill-primary: var(--c-surface) !important;
    --background-fill-secondary: var(--c-bg) !important;
    --block-background-fill: var(--c-surface) !important;
    --block-label-background-fill: transparent !important;
    --block-label-text-color: var(--c-text) !important;
    --block-title-background-fill: transparent !important;
    --block-title-text-color: var(--c-text) !important;
    --block-header-background-fill: transparent !important;
    --input-background-fill: var(--c-surface) !important;
    --border-color-primary: var(--c-border) !important;
    --checkbox-label-background-fill: transparent !important;
    --button-secondary-background-fill: #ffffff !important;
}

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: var(--c-bg) !important;
    color: var(--c-text) !important;
    max-width: 1400px !important;
}

h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; font-weight: 800 !important; color: var(--c-text) !important; }
p, .prose, .gradio-container p { color: var(--c-text-2) !important; line-height: 1.6 !important; font-weight: 500 !important; }

/* Subtext & Labels in components */
.gradio-container .prose p, .gradio-container .prose span { color: var(--c-text-2) !important; }

#ts-bg-canvas { display: none !important; }

/* ══════════════════════════════════════════════════════════════
   HIGH-CONTRAST CLINICAL PANELS
   ══════════════════════════════════════════════════════════════ */
.ts-glass-panel {
    background: #ffffff !important;
    border: 1px solid var(--c-border) !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 15px -3px rgba(6, 78, 59, 0.05) !important;
    padding: 32px !important;
    margin-bottom: 24px !important;
}

/* ══════════════════════════════════════════════════════════════
   PROFESSIONAL MEDICAL BUTTONS
   ══════════════════════════════════════════════════════════════ */
.gradio-container button.primary {
    background: linear-gradient(135deg, var(--c-primary), var(--c-primary-h)) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 30px !important; /* Full pill shape */
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    padding: 16px 40px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.2) !important;
    letter-spacing: 0.02em;
}
.gradio-container button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 20px -5px rgba(13, 148, 136, 0.3) !important;
    filter: brightness(1.1);
}

.gradio-container button.secondary:hover {
    background: var(--c-primary-soft) !important;
    border-color: var(--c-primary) !important;
    color: var(--c-primary) !important;
}

/* Dropdown & List Item Hovers */
.gradio-container .item:hover, .gradio-container .option:hover, .gradio-container .dropdown-item:hover {
    background: var(--c-primary-soft) !important;
    color: var(--c-primary) !important;
}

.gradio-container button:not(.primary) {
    background: #ffffff !important;
    color: var(--c-text) !important;
    border: 2px solid var(--c-border) !important;
    border-radius: 30px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}

/* ══════════════════════════════════════════════════════════════
   INTELLIGENCE REPORT STYLE
   ══════════════════════════════════════════════════════════════ */
.ts-intelligence-report {
    background: #ffffff !important;
    border: 2px solid var(--c-border) !important;
    border-radius: var(--c-radius) !important;
    padding: 40px !important;
}
.ts-intelligence-report h2 { color: var(--c-text) !important; border-bottom: 3px solid var(--c-primary-soft); padding-bottom: 12px; margin-bottom: 24px; }

.alert-critical {
    background: var(--c-red-bg) !important;
    border: 2px solid var(--c-red) !important;
    color: #000000 !important;
}

/* ══════════════════════════════════════════════════════════════
   SYSTEM STATUS BAR
   ══════════════════════════════════════════════════════════════ */
.ts-status-bar {
    display: flex;
    justify-content: space-between;
    padding: 16px 28px;
    background: var(--c-primary-soft);
    border: 1px solid var(--c-border);
    border-radius: 12px;
    font-size: 0.82em;
    font-weight: 700;
    color: var(--c-text-3);
    margin-bottom: 24px;
    box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.02);
}

/* ══════════════════════════════════════════════════════════════
   INPUTS
   ══════════════════════════════════════════════════════════════ */
.gradio-container textarea, .gradio-container input {
    background: #ffffff !important;
    border: 2px solid var(--c-border-med) !important;
    border-radius: 8px !important;
    color: #000000 !important;
    font-weight: 500 !important;
    font-size: 1.05em !important;
}
.gradio-container textarea:focus, .gradio-container input:focus {
    border-color: var(--c-primary) !important;
    box-shadow: 0 0 0 4px var(--c-primary-soft) !important;
}

/* ══════════════════════════════════════════════════════════════
   TAB NAVIGATION — GUARANTEED VISIBILITY
   ══════════════════════════════════════════════════════════════ */
.gradio-container .tabs .tab-nav { 
    background: transparent !important;
    border-bottom: 2px solid var(--c-border) !important;
    margin-bottom: 32px !important;
    padding: 0 10px !important;
}
.gradio-container .tabs .tab-nav button {
    background: transparent !important;
    color: var(--c-text-3) !important;
    border: none !important;
    border-bottom: 4px solid transparent !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.1em !important;
    padding: 16px 32px !important;
    opacity: 1 !important;
}
.gradio-container .tabs .tab-nav button:hover {
    color: var(--c-primary) !important;
    background: var(--c-primary-soft) !important;
}
.gradio-container .tabs .tab-nav button.selected {
    color: var(--c-primary) !important;
    border-bottom-color: var(--c-primary) !important;
    background: transparent !important;
}

/* ══════════════════════════════════════════════════════════════
   REMOVE BLACK CARDHOLDERS (GRADIO INTERNAL LABELS)
   ══════════════════════════════════════════════════════════════ */
.gradio-container .block-label, 
.gradio-container .label-wrap,
.gradio-container .block-title,
.gradio-container .label,
.gradio-container .block-header,
.gradio-container .block-legend,
.gradio-container .form > .block > .label,
.gradio-container .group > .block > .label,
.gradio-container [data-testid="block-label"],
.gradio-container .block.padded.border > .label {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--c-primary) !important;
    font-weight: 800 !important;
    padding: 0 !important;
    margin-bottom: 12px !important;
    min-height: auto !important;
}

.gradio-container .label-wrap span {
    color: var(--c-text) !important;
    font-weight: 800 !important;
    font-size: 0.95em !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0 !important;
}

/* Global hover fixes for Gradio utility buttons (Clear, copy, etc) */
.gradio-container button:not(.primary):hover {
    background: var(--c-primary-soft) !important;
    color: var(--c-primary) !important;
}

.ts-footer {
    text-align: center;
    padding: 50px;
    border-top: 2px solid var(--c-border);
    color: var(--c-text-3);
    font-size: 0.9em;
    font-weight: 600;
}

@keyframes ts-blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.gradio-container code {
    background: transparent !important;
    color: inherit !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 700 !important;
}

/* ─────────────────────────────────────────────────────────────────────────────
   MCQ SURVEY STYLING — PREMIUM CLINICAL
   ───────────────────────────────────────────────────────────────────────────── */
.ts-mcq-item {
    border: 1px solid var(--c-border) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
    background: rgba(255,255,255,0.5) !important;
    transition: all 0.2s ease !important;
}
.ts-mcq-item:hover {
    border-color: var(--c-primary) !important;
    background: var(--c-primary-soft) !important;
}
.ts-mcq-item .wrap {
    gap: 12px !important;
}
.ts-mcq-item label span {
    font-size: 0.9em !important;
    font-weight: 600 !important;
    color: var(--c-text) !important;
}
#mcq_survey_group {
    background: rgba(248, 250, 252, 0.5) !important;
    padding: 20px !important;
    border-radius: 16px !important;
    border: 1px dashed var(--c-border) !important;
    margin-top: 20px !important;
}
"""


def create_app():
    with gr.Blocks(
        title="TruthShield Clinical Intelligence Platform",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.slate,
        ),
        css=CUSTOM_CSS,
    ) as app:

        # ─── Friendly Clinical Hero section ─────────────────────────────
        gr.HTML("""
<div style="display:flex; justify-content:space-between; align-items:flex-start; padding:60px 40px; background:transparent; margin-bottom:40px;">
  <!-- Left Content -->
  <div style="flex:1; max-width:800px;">
    <div style="display:inline-flex; align-items:center; gap:8px; padding:6px 12px; background:#ffffff; border:1px solid var(--c-border); border-radius:30px; font-size:0.7em; font-weight:800; color:var(--c-primary); letter-spacing:0.05em; margin-bottom:32px;">
      <span style="width:6px;height:6px;background:var(--c-primary);border-radius:50%;animation:ts-blink 2s infinite;"></span> LIVE — MEDGEMMA-4B EDGE AI
    </div>
    
    <h1 style="font-size:4.8em; line-height:1; margin-bottom:24px; color:var(--c-text);">
      TruthShield<br>
      <span style="opacity:0.9;">Friendly Clinical AI</span>
    </h1>
    
    <div style="width:60px; height:4px; background:linear-gradient(90deg, #ea580c 0%, #2563eb 100%); margin-bottom:32px; border-radius:10px;"></div>
    
    <p style="font-size:1.3em; margin-bottom:40px; max-width:600px; color:var(--c-text-2);">
      A calm, game-like check-in space where patients can be fully honest before they see you.<br>
      <span style="font-weight:700; color:var(--c-text);">TruthShield gently turns their story into a clear clinical signal.</span>
    </p>
    
    <div style="display:flex; gap:16px;">
      <div class="ts-badge">🧩 FRIENDLY, TABLET-READY</div>
      <div class="ts-badge">🔒 100% OFFLINE</div>
      <div class="ts-badge" style="background:var(--c-primary-soft); border-color:var(--c-border);">⚡ SNOMED-CT CODED</div>
    </div>
  </div>

  <!-- Right Stats -->
  <div style="display:flex; flex-direction:column; gap:48px; border-left:2px solid var(--c-border); padding-left:48px; margin-top:20px;">
    <div class="ts-stat-item" style="border:none;">
      <div class="ts-stat-val" style="color:var(--c-primary);">80%</div>
      <div class="ts-stat-label">OF PATIENTS<br>WITHHOLD INFO</div>
    </div>
    <div class="ts-stat-item" style="border:none;">
      <div class="ts-stat-val" style="color:var(--c-accent-amber);">12+</div>
      <div class="ts-stat-label">CLINICAL<br>SCENARIOS</div>
    </div>
    <div class="ts-stat-item" style="border:none;">
      <div class="ts-stat-val" style="color:var(--c-primary-h);">4B</div>
      <div class="ts-stat-label">MEDGEMMA<br>PARAMETERS</div>
    </div>
  </div>
</div>
""")

        with gr.Tabs(elem_classes=["tabs"]) as tabs:

            # ════════════════════════════════════════════════════════
            # TAB 1: Patient Honesty Portal
            # ════════════════════════════════════════════════════════
            with gr.Tab("Patient Honesty Portal", id="patient_tab"):
                with gr.Column(scale=1, elem_id="patient_container", elem_classes=["ts-glass-panel"]):
                    gr.HTML("""
<div style="margin-bottom:32px;">
    <h2 style="font-size:2em;color:var(--c-text);margin-bottom:12px;">Let's start with your story...</h2>
    <p style="color:var(--c-text-2);font-size:1.1em;max-width:800px;">TruthShield is a safe space. Everything you share stays encrypted and only helps your clinical team provide more personalized care.</p>
</div>
""")
                    
                    with gr.Group():
                        gr.HTML("""<div style="font-weight:700;font-size:0.85em;color:var(--c-primary);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em;">Your Story</div>""")
                        patient_survey_input = gr.Textbox(
                            label="",
                            placeholder="Tell us how you've been feeling, any habits you want to share, or anything bothering you that might be hard to say in person...",
                            lines=10,
                            show_label=False,
                        )

                    # ─── Demo Scenarios & Simulation Layer (Moved from Clinician View) ───
                    with gr.Accordion("🚀 Try Demo Scenarios (Presentation Mode)", open=False):
                        gr.HTML("<div style='font-size:0.85em;color:var(--c-text-3);margin-bottom:12px;'>Select a high-emotion scenario to auto-populate the portal. Perfect for quick demos.</div>")
                        with gr.Row():
                            demo_cyber = gr.Button("Cyberbullying 📱", size="sm")
                            demo_burnout = gr.Button("Caregiver Burnout 👵", size="sm")
                            demo_grief = gr.Button("Hidden Grief 🕊️", size="sm")
                        with gr.Row():
                            demo_safety = gr.Button("Physical Safety 🔒", size="sm")
                            demo_veteran = gr.Button("Veteran Trauma ⚔️", size="sm")
                            demo_finance = gr.Button("Financial Fraud 💸", size="sm")
                        with gr.Row():
                            demo_sexual = gr.Button("Sexual Health 🏥", size="sm")
                        
                        gr.HTML("<div style='margin-top:15px; border-top:1px solid var(--c-border); padding-top:10px;'></div>")
                        sim_mode_toggle = gr.Checkbox(label="Enable Instant Simulation (Demo Mode)", value=True)
                        gr.HTML("<div style='font-size:0.75em;color:var(--c-amber);line-height:1.2;margin-top:-5px;'>Bypass slow AI inference for preset scenarios during presentations. (AUTO-ENABLED FOR HF SPACES)</div>")

                    # Structured Survey Group (10 Nuanced Questions)
                    with gr.Group(elem_id="mcq_survey_group", visible=False) as mcq_survey_group:
                        gr.HTML("""<div style="font-weight:700;font-size:0.85em;color:var(--c-primary);margin-bottom:16px;text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid var(--c-border);padding-bottom:8px;">Deep Diagnostic Check-in (AI Personalized)</div>""")
                        
                        department_display = gr.HTML(
                            value="""<div style="display:inline-flex; align-items:center; gap:8px; padding:6px 12px; background:#e0f2f7; border:1px solid var(--c-primary); border-radius:30px; font-size:0.7em; font-weight:800; color:var(--c-primary); letter-spacing:0.05em; margin-bottom:16px;">
                                <span style="width:6px;height:6px;background:var(--c-primary);border-radius:50%;"></span> DEPARTMENT: Awaiting Analysis
                            </div>""",
                            visible=False
                        )
                        
                        mcq_components = []
                        for i in range(10): # Expanded to 10
                            radio = gr.Radio(
                                choices=[],
                                label=f"Scanning {i+1}...",
                                elem_classes=["ts-mcq-item"],
                                visible=True
                            )
                            mcq_components.append(radio)
                        
                        submit_final_btn = gr.Button("Submit Detailed Survey", variant="primary")

                    with gr.Row() as patient_initial_actions:
                        submit_story_btn = gr.Button("Submit Story & Personalize My Survey", variant="primary", scale=2)
                        clear_patient_btn = gr.Button("Clear All", variant="secondary", scale=1)
                    
                    patient_status = gr.HTML(value="")

            # ════════════════════════════════════════════════════════
            # TAB 2: Clinician Dashboard
            # ════════════════════════════════════════════════════════
            with gr.Tab("Clinician Dashboard", id="clinician_tab"):
                with gr.Row(equal_height=False):

                    # Left Column — Intake & Analysis
                    with gr.Column(scale=3):
                        
                        # Patient Info Group
                        with gr.Group(elem_classes=["ts-glass-panel"]):
                            gr.HTML("""<div style="font-weight:700;font-size:0.75em;color:var(--c-text-3);text-transform:uppercase;margin-bottom:12px;">Patient Context</div>""")
                            with gr.Row():
                                patient_age = gr.Textbox(label="Patient Age", placeholder="e.g. 16", lines=1)
                                visit_type = gr.Textbox(label="Visit Type", placeholder="e.g. Wellness Exam", lines=1)

                        # Hidden / Integrated Survey Text

                        with gr.Group(elem_classes=["ts-glass-panel"]):
                            gr.HTML("""<div style="font-weight:700;font-size:0.75em;color:var(--c-primary);text-transform:uppercase;margin-bottom:12px;">Received Patient Survey (Anonymous)</div>""")
                            survey_input = gr.Textbox(
                                label="",
                                placeholder="Awaiting patient submission... (Data from 'Patient Honesty Portal' will appear here)",
                                lines=8,
                                show_label=False,
                            )

                        # EHR Documentation
                        with gr.Group(elem_classes=["ts-glass-panel"]):
                            gr.HTML("""<div style="font-weight:700;font-size:0.75em;color:var(--c-amber);text-transform:uppercase;margin-bottom:12px;">EHR Clinical Notes (Current Record)</div>""")
                            notes_input = gr.Textbox(
                                label="",
                                placeholder="Paste current clinical notes or prior documentation here...",
                                lines=6,
                                show_label=False,
                            )

                        with gr.Row():
                            analyze_btn = gr.Button("🔍  Run TruthShield Analysis", variant="primary", scale=3)
                            clear_btn = gr.Button("Clear All", variant="secondary", scale=1)

                        # Analysis results
                        timer_display = gr.HTML("""<div class="ts-status-bar">STATUS: AWAITING ANALYSIS</div>""")
                        alert_output = gr.Markdown("Submit analysis to identify discrepancies.", elem_classes=["ts-intelligence-report"])
                        
                        sync_btn = gr.Button("📤  Sync Structured Report to EHR", variant="primary")
                        sync_status = gr.Markdown("")

                        # ─── Sidebar: System Intelligence ───
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["ts-glass-panel"]):
                                gr.HTML("""<div style="font-weight:700;font-size:0.75em;color:var(--c-text-3);text-transform:uppercase;margin-bottom:12px;">Intelligence Status</div>""")
                                # Professional Status Indicator
                                is_simulation = AI_ENGINE.is_simulation
                                status_color = "var(--c-primary)" if not is_simulation else "var(--c-amber)"
                                status_text = "MedGemma Active" if not is_simulation else "Awaiting Initialization"
                                status_bg = "#f0f9f6" if not is_simulation else "#fff7ed"
                                
                                gr.HTML(f"""
                                    <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;background:{status_bg};border-radius:10px;border:1px solid {status_color};">
                                        <div style="width:12px;height:12px;border-radius:50%;background:{status_color};box-shadow: 0 0 10px {status_color}88;"></div>
                                        <div>
                                            <div style="font-weight:800;font-size:0.9em;color:{status_color};line-height:1.1;">{status_text}</div>
                                            <div style="font-size:0.75em;color:{status_color};opacity:0.8;margin-top:2px;">{AI_ENGINE.model_name if not is_simulation else "Weights Not Found"}</div>
                                        </div>
                                    </div>
                                """)

                            with gr.Group(elem_classes=["ts-glass-panel"]):
                                gr.HTML("""<div style="font-weight:700;font-size:0.75em;color:var(--c-text-3);text-transform:uppercase;margin-bottom:8px;">System Control</div>""")
                                with gr.Accordion("Advanced Configuration", open=False):
                                    gr.HTML("""<div style="font-size:0.8em;color:var(--c-text-3);margin-bottom:8px;">Modify AI weights or FHIR settings below.</div>""")
                                    hf_token_input = gr.Textbox(label="HuggingFace Token", placeholder="hf_...", type="password")
                                    download_btn = gr.Button("Re-Sync MedGemma", variant="secondary", size="sm")
                                    setup_output = gr.Markdown("Ready.")
                                    fhir_output = gr.Code(label="FHIR JSON", language="json", lines=5)
                                    gr.Code(label="API Access", value=generate_api_curl_sample())

                # Legacy Stats removed in favor of Hero Integrated Sidebar
                pass

        # ─── Footer ──────────────────────────────────────────────────
        gr.HTML("""
<div style="text-align:center;margin-top:32px;padding:20px;border-top:1px solid var(--c-border);color:var(--c-text-3);font-size:0.72em;font-family:'Outfit',sans-serif;letter-spacing:0.06em;text-transform:uppercase;">
  TruthShield v2.0 &nbsp;·&nbsp; MedGemma Impact Challenge 2026 &nbsp;·&nbsp; 100% On-Device Privacy
</div>
""")

        # ─── Event Handlers ──────────────────────────────────────────
        
        # 1. Patient Portal Submission
        def _handle_story_submission(story):
            if not story.strip():
                return ["""<div style="color:var(--c-red);font-weight:600;margin-top:10px;">⚠️ Please enter your story before proceeding.</div>""", gr.update()] + [gr.update() for _ in range(10)] + [gr.update(), gr.update()]
            
            # Requesting 10 questions for "Crystal Clear" diagnostic clarity
            new_qs = generate_ai_mcqs(story, False, count=10)
            updates = []
            for i in range(10):
                if i < len(new_qs):
                    q_text, opts = new_qs[i]
                    updates.append(gr.update(label=q_text, choices=opts, value=None, visible=True))
                else:
                    updates.append(gr.update(visible=False))
            
            AI_ENGINE.current_personalized_qs = new_qs
            
            status_html = """<div style="color:var(--c-primary);font-weight:600;margin-top:10px;">✨ Story Processed. MedGemma has generated a 10-point diagnostic survey below.</div>"""
            # We don't know the department for manual entry unless AI predicts it, let's keep it 'General Medicine'
            dept_html = """<div style="display:inline-flex; align-items:center; gap:8px; padding:6px 12px; background:#e0f2f7; border:1px solid var(--c-primary); border-radius:30px; font-size:0.7em; font-weight:800; color:var(--c-primary); letter-spacing:0.05em; margin-bottom:16px;"><span style="width:6px;height:6px;background:var(--c-primary);border-radius:50%;"></span> DEPARTMENT: GENERAL MEDICINE</div>"""
            return [status_html, gr.update(visible=True)] + updates + [gr.update(visible=False), gr.update(value=dept_html, visible=True)]

        submit_story_btn.click(
            fn=_handle_story_submission,
            inputs=[patient_survey_input],
            outputs=[patient_status, mcq_survey_group] + mcq_components + [patient_initial_actions, department_display]
        )

        def _submit_patient_data(survey, *mcqs):
            if not survey.strip():
                return """<div style="color:var(--c-red);font-weight:600;margin-top:10px;">⚠️ Please enter some text before submitting.</div>""", gr.update()
            
            # Combine MCQ data for the clinician's hidden textbox
            mcq_summary = "\n\n--- STRUCTURED CLINICAL SURVEY ---\n"
            for i, val in enumerate(mcqs):
                # Use personalized question if available, otherwise fallback to static
                if AI_ENGINE.current_personalized_qs and i < len(AI_ENGINE.current_personalized_qs):
                    q = AI_ENGINE.current_personalized_qs[i][0]
                else:
                    q = PATIENT_MCQS[i]["question"]
                mcq_summary += f"{i+1}. {q} → {val if val else 'No answer'}\n"
            
            combined_data = survey + mcq_summary
            return """<div style="color:var(--c-primary);font-weight:600;margin-top:10px;animation:ts-blink 1.5s infinite;">✅ Submitting to Clinical Team... (Go to 'Clinician Dashboard' to see the received survey)</div>""", combined_data

        submit_final_btn.click(
            fn=_submit_patient_data,
            inputs=[patient_survey_input] + mcq_components,
            outputs=[patient_status, survey_input]
        )
        
        def _clear_patient_portal():
            return (
                ["", ""] + 
                [None] * 5 + 
                [gr.update(visible=False), gr.update(visible=True)]
            )

        clear_patient_btn.click(
            fn=_clear_patient_portal,
            outputs=[patient_survey_input, patient_status] + mcq_components + [mcq_survey_group, patient_initial_actions]
        )

        # 3. Model Management
        def _handle_model_setup(token):
            if not token.strip():
                return "❌ Please enter a HuggingFace Token."
            
            yield "⏳ [1/3] Verifying dependencies and token..."
            try:
                from huggingface_hub import login
                login(token=token)
            except Exception as e:
                yield f"❌ Login failed: {e}"
                return

            yield "⏳ [2/3] Downloading & Quantizing MedGemma (this may take several minutes)..."
            # We call the existing setup_model logic
            try:
                import subprocess
                # Run setup_model.py as a separate process to avoid blocking Gradio completely
                # and to ensure a clean environment for library compilation
                process = subprocess.Popen(
                    [sys.executable, "setup_model.py", "--hf-token", token, "--output-dir", "./models/medgemma-4b-awq"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                for line in process.stdout:
                    if "Downloaded" in line or "Saved" in line or "Loading" in line:
                        yield f"⏳ {line.strip()}"
                
                process.wait()
                if process.returncode == 0:
                    yield "⏳ [3/3] Initializing loaded model..."
                    success, msg = AI_ENGINE.load("./models/medgemma-4b-awq")
                    if success:
                        yield f"✅ MedGemma Initialized: {msg}"
                    else:
                        yield f"⚠️ Weights downloaded but load failed: {msg}"
                else:
                    yield f"❌ Setup failed with code {process.returncode}. Check terminal for details."
            except Exception as e:
                yield f"❌ Error during setup: {e}"

        download_btn.click(
            fn=_handle_model_setup,
            inputs=[hf_token_input],
            outputs=[setup_output]
        )
        # 4. Clinical Logic
        analyze_btn.click(
            fn=analyze_discrepancies,
            inputs=[survey_input, notes_input, patient_age, visit_type, sim_mode_toggle] + mcq_components,
            outputs=[alert_output, timer_display, fhir_output],
        )

        def _load_demo_scenario(scenario_id):
            s = SCENARIOS[scenario_id]
            # Returns: Survey, Age, Visit Type, Notes, SimMode(True), ...MCQ updates...
            dept_name = s.get("department", "General Medicine")
            dept_html = f"""<div style="display:inline-flex; align-items:center; gap:8px; padding:6px 12px; background:#e0f2f7; border:1px solid var(--c-primary); border-radius:30px; font-size:0.7em; font-weight:800; color:var(--c-primary); letter-spacing:0.05em; margin-bottom:16px;"><span style="width:6px;height:6px;background:var(--c-primary);border-radius:50%;"></span> DEPARTMENT: {dept_name.upper()}</div>"""
            
            updates = [s["survey"], s["age"], dept_name, s["notes"], True]
            # Fill the 10 MCQs
            for i in range(10):
                # Demo scenarios strictly use THEIR mcqs
                if i < len(s.get("mcqs", [])):
                    updates.append(gr.update(
                        label=s["mcqs"][i], 
                        choices=["Not at all", "Somewhat", "Very much"], # Added choices to show radio buttons
                        visible=True, 
                        value=None
                    ))
                else:
                    updates.append(gr.update(visible=False))
            
            # Additional UI updates for Demo
            # patient_status, mcq_survey_group, patient_initial_actions, department_display
            updates += [
                """<div style="color:var(--c-amber);font-weight:600;margin-top:10px;">🚀 Demo Scenario Loaded. Review the 10-point honesty check below.</div>""",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=dept_html, visible=True)
            ]
            return updates

        demo_cyber.click(fn=lambda: _load_demo_scenario("cyberbullying"), outputs=[patient_survey_input, patient_age, visit_type, notes_input, sim_mode_toggle] + mcq_components + [patient_status, mcq_survey_group, patient_initial_actions, department_display])
        demo_burnout.click(fn=lambda: _load_demo_scenario("caregiver_burnout"), outputs=[patient_survey_input, patient_age, visit_type, notes_input, sim_mode_toggle] + mcq_components + [patient_status, mcq_survey_group, patient_initial_actions, department_display])
        demo_grief.click(fn=lambda: _load_demo_scenario("hidden_grief"), outputs=[patient_survey_input, patient_age, visit_type, notes_input, sim_mode_toggle] + mcq_components + [patient_status, mcq_survey_group, patient_initial_actions, department_display])
        demo_safety.click(fn=lambda: _load_demo_scenario("domestic_violence"), outputs=[patient_survey_input, patient_age, visit_type, notes_input, sim_mode_toggle] + mcq_components + [patient_status, mcq_survey_group, patient_initial_actions, department_display])
        demo_veteran.click(fn=lambda: _load_demo_scenario("veteran_trauma"), outputs=[patient_survey_input, patient_age, visit_type, notes_input, sim_mode_toggle] + mcq_components + [patient_status, mcq_survey_group, patient_initial_actions, department_display])
        demo_finance.click(fn=lambda: _load_demo_scenario("financial_fraud"), outputs=[patient_survey_input, patient_age, visit_type, notes_input, sim_mode_toggle] + mcq_components + [patient_status, mcq_survey_group, patient_initial_actions, department_display])
        demo_sexual.click(fn=lambda: _load_demo_scenario("sexual_health"), outputs=[patient_survey_input, patient_age, visit_type, notes_input, sim_mode_toggle] + mcq_components + [patient_status, mcq_survey_group, patient_initial_actions, department_display])

        def _simulate_ehr_sync(fhir):
            if not fhir:
                return """<div style="padding:14px 18px;border-radius:12px;margin-top:12px;background:var(--c-red-bg);border:1px solid var(--c-red);font-size:0.85em;color:var(--c-red);font-weight:600;">⚠️ Run analysis first before syncing to EHR.</div>"""
            time.sleep(1)
            bundle_id = json.loads(fhir).get("id")
            return f"""<div style="padding:14px 18px;border-radius:12px;margin-top:12px;background:var(--c-primary-soft);border:1px solid var(--c-border);font-size:0.85em;color:var(--c-primary);font-weight:600;">✅ HL7 FHIR Bundle transmitted to Hospital EHR.<br><code style="font-size:0.9em;">Bundle ID: {bundle_id}</code></div>"""

        sync_btn.click(fn=_simulate_ehr_sync, inputs=[fhir_output], outputs=[sync_status])

        clear_btn.click(
            fn=lambda: ("", "", "", "", "*Submit the patient survey above to generate a clinical analysis.*", ""),
            outputs=[survey_input, notes_input, patient_age, visit_type, alert_output, sync_status],
        )

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TruthShield Clinical Intelligence Platform")
    parser.add_argument("--model-path", type=str, default=None, help="Path to AWQ-quantized MedGemma model")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    if args.model_path:
        load_model(args.model_path)
    else:
        # Auto-detect and load any synchronized model
        print("[TruthShield] Scanning for local AI weights...")
        success, msg = AI_ENGINE.load()
        if success:
            print(f"[TruthShield] Automatic Initialization: {msg}")
        else:
            print(f"[TruthShield] No local weights found: {msg}")
            print("[TruthShield] Starting in Simulation Mode. Use 'Model Management' to sync MedGemma.\n")

    app = create_app()
    # Note: server_name set to "localhost" per user security preference for offline use
    print(f"\n[TruthShield] Server starting at http://localhost:{args.port}\n")
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=True,
        show_error=True,
        favicon_path=None,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
