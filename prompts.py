"""
TruthShield — Prompt Templates for MedGemma Clinician Alert Generation

These prompts are engineered for MedGemma-4B-instruct to produce structured,
non-confrontational clinical discrepancy alerts. The prompts use evidence-based
framing and cite sensitivity of topics to maximize clinical utility.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Sets MedGemma's role and output contract
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are TruthShield Clinical AI. Your role is identify potentially life-threatening or financially high-risk discrepancies.
CRITICAL: BE EXTREMELY CONCISE. Avoid long explanations. Frame discrepancies as "opportunities for conversation."
"""


# ─────────────────────────────────────────────────────────────────────────────
# DISCREPANCY ANALYSIS PROMPT — Compares survey vs. clinical notes
# ─────────────────────────────────────────────────────────────────────────────

DISCREPANCY_ANALYSIS_PROMPT = """Analyze the survey vs. clinical notes. Identify ALL discrepancies. 
BE EXTREMELY BRIEF. Use a simple list.

## Anonymous Patient Survey
{survey_responses}

## Prior Clinical Notes
{clinical_notes}

## Instructions
For each discrepancy found, provide ONLY:
1. **Category**
2. **Fact Mapping**: (Survey says vs. Notes say)
3. **Severity**: CRITICAL | HIGH | MODERATE
4. **Reasoning**: (One short sentence)
5. **Approach**: (Short MI opener)

Keep the entire output under 150 tokens.
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLINICIAN ALERT PROMPT — Generates the final compassionate alert
# ─────────────────────────────────────────────────────────────────────────────

CLINICIAN_ALERT_PROMPT = """Based on the discrepancy analysis below, generate a concise, actionable CLINICIAN ALERT suitable for display in a clinical dashboard. The alert must be:

- Non-confrontational and patient-centered
- Prioritized by severity (CRITICAL items first)
- Actionable with specific conversation starters
- Brief enough to read in under 60 seconds

## Discrepancy Analysis
{discrepancy_analysis}

## Patient Context
Age: {patient_age}
Visit Type: {visit_type}

Generate the alert now. Begin with the highest-severity item. For each item include:
🔴/🟡/🟢 [SEVERITY] — [CATEGORY]
• Discrepancy: [brief description]
• Suggested approach: "[exact words the clinician could say]"
• Clinical reasoning: [one sentence on why this matters]

End with a summary line: "X discrepancies detected (Y critical, Z high, W moderate)."
"""


# ─────────────────────────────────────────────────────────────────────────────
# MCQ GENERATION PROMPT — Generates 10 custom screening questions
# ─────────────────────────────────────────────────────────────────────────────

MCQ_GENERATION_PROMPT = """You are a clinical psychometrician. Based on the following patient story, generate EXACTLY 10 deep clinical questions to identify masked truths or discrepancies.

## Patient Story
{patient_story}

## Formatting Requirements:
1. Generate EXACTLY 10 questions for "crystal clear" clinical clarity.
2. For each question, provide 3 clinical options.
3. Be EXTREMELY CONCISE but clinically rigorous.
4. Format: EXACTLY one question per line.
5. Format per line: [Number]. [Question] | [Opt1], [Opt2], [Opt3]

## Example:
1. How is your appetite? | Normal, Reduced, Increased
2. Do you feel safe at home? | Yes, No, Uncertain

Generate 10 SHORT clinical MCQs now.
"""


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION PROMPT — For instant demo mode
# ─────────────────────────────────────────────────────────────────────────────

def get_simulated_alert(scenario_id: str) -> str:
    """Returns a pre-generated clinician alert for demo/simulation mode."""

    SIMULATED_ALERTS = {
        "cyberbullying": """## 🚨 TruthShield Clinical Alert — HIGH
**Patient**: Anonymous (14 y/o, wellness check)
**Discrepancies Detected**: 2 (0 Critical, 2 High)
**Generated by**: MedGemma-4B (Simulation Engine)

---

### 🟡 HIGH — Social Trauma / Cyberbullying
• **Discrepancy**: Patient anonymously reported severe distress due to "everyone at school hating me" and constant bullying via phone. Clinical notes and parent report describe patient as merely "moody" or "withdrawn" (typical adolescent adjustment).
• **Suggested approach**: *"It's really common for things happening online to affect how we feel in real life. I've seen a lot of people your age dealing with some pretty mean stuff on social media. Has anything like that been on your mind lately?"*
• **Clinical reasoning**: Persistent cyberbullying is a high-risk factor for depressive disorders and self-harm in adolescents. The physical manifestation (stomach aches) suggests a high level of somatized stress.

### 🟡 HIGH — Somatic Stress / Sleep Disturbance
• **Discrepancy**: Patient anonymously reported daily stomach pain and inability to sleep due to phone notifications. Clinical notes only mention being "tired."
• **Suggested approach**: *"I want to talk more about those stomach aches and your sleep. Sometimes when we're really stressed or worried, our bodies feel it first. Does it feel like your phone or school is getting in the way of your rest?"*
• **Clinical reasoning**: Chronic sleep deprivation and somatization in a 14-year-old require targeted intervention.

---
**Summary**: 2 discrepancies detected. Clear evidence of cyberbullying impacting physical and mental health. **Recommend private session with patient and assessment for social media-related anxiety/depression.**""",

        "caregiver_burnout": """## 🚨 TruthShield Clinical Alert — HIGH
**Patient**: Anonymous (52 y/o, follow-up)
**Discrepancies Detected**: 3 (0 Critical, 1 High, 2 Moderate)
**Generated by**: MedGemma-4B (Simulation Engine)

---

### 🟡 HIGH — Mental Health / Caregiver Crisis
• **Discrepancy**: Patient anonymously reported extreme resentment and guilt related to dementia care, with <4h sleep for months. Clinical notes state patient is "managing."
• **Suggested approach**: *"Caring for someone with dementia is one of the hardest jobs there is. Many people feel completely burnt out or even resentful, and that's okay to admit. How are you really holding up with your mother's care?"*
• **Clinical reasoning**: Extreme caregiver burden is a precursor to clinical depression and health collapse for the caregiver. Resentment is a primary indicator of burnout.

### 🟢 MODERATE — Physical Health / Somatic Pain
• **Discrepancy**: Patient anonymously reported "back killing me" and daily migraines. Notes mention "mechanical back pain."
• **Suggested approach**: *"We've been treating the back pain as physical, but I wonder if the stress of the 24/7 care you're providing is making it worse. Do the migraines seem to spike when your mother's symptoms are worse?"*
• **Clinical reasoning**: The patient's physical symptoms are likely exacerbated by chronic cortisol elevation from sleep deprivation and stress.

---
**Summary**: 3 discrepancies detected. Patient is at a breaking point due to caregiver burnout. **Recommend referral to adult daycare services for the mother and counseling/support groups for the patient.**""",

        "hidden_grief": """## 🚨 TruthShield Clinical Alert — CRITICAL
**Patient**: Anonymous (65 y/o, post-CVA)
**Discrepancies Detected**: 3 (1 Critical, 1 High, 1 Moderate)
**Generated by**: MedGemma-4B (Simulation Engine)

---

### 🔴 CRITICAL — Medication Non-Adherence / Risk
• **Discrepancy**: Patient anonymously reported stopping BP meds because "I just don't care anymore" after partner's death. Notes document patient as "adherent" and stable.
• **Suggested approach**: *"I noticed your blood pressure is up today. I know you said you're taking your meds, but sometimes when we're going through a huge life change, it's hard to stay on top of things—or we might even wonder if it matters. How have you been feeling since your partner passed?"*
• **Clinical reasoning**: Abrupt cessation of post-CVA prophylactic medication in a grieving patient is a high stroke risk and a marker for "silent" depression.

---
**Summary**: 3 search-driven discrepancies. Grieving process is severely impacting medical compliance. **Immediate psychiatric evaluation for bereavement-related depression and restart of cardiovascular regimen required.**""",

        "domestic_violence": """## 🚨 TruthShield Clinical Alert — CRITICAL
**Patient**: Anonymous (28 y/o, injury)
**Discrepancies Detected**: 2 (1 Critical, 1 High)
**Generated by**: MedGemma-4B (Simulation Engine)

---

### 🔴 CRITICAL — Physical Safety / IPV
• **Discrepancy**: Patient anonymously reported "terror" and being pushed by partner. Notes record "tripped over dog" with partner present.
• **Suggested approach**: *"I'd like to do a specialized physical exam in private. Standard policy for this type of injury is for the clinician to speak with the patient alone for a few minutes. I'll ask your partner to wait in the hall."*
• **Clinical reasoning**: Mismatch between "accidental fall" and reported pushing is pathognomonic for Interpersonal Violence. High risk for escalating harm.

---
**Summary**: 2 discrepancies. Confirmed concealement of violence. **Mandatory private safety assessment and IPV resource provision.**""",

        "veteran_trauma": """## 🚨 TruthShield Clinical Alert — HIGH
**Patient**: Anonymous (34 y/o, veteran)
**Discrepancies Detected**: 3 (0 Critical, 2 High, 1 Moderate)
**Generated by**: MedGemma-4B (Simulation Engine)

---

### 🟡 HIGH — PTSD / Avoidance
• **Discrepancy**: Patient anonymously reported extreme hyper-vigilance and panic in stores. Notes show standardized screen was "0" (negative).
• **Suggested approach**: *"A lot of veterans find that the screens we use in the clinic don't really capture the daily reality of being back. Have you felt that 'on-guard' feeling lately, maybe in crowded places?"*
• **Clinical reasoning**: Negative screens often reflect stoic masking or lack of trust in standard tools.

---
**Summary**: 3 discrepancies. Masked PTSD symptoms with self-medication (alcohol). **Recommend referral to specialized veteran reintegration services.**""",

        "financial_fraud": """## 🚨 TruthShield Clinical Alert — CRITICAL
**Patient**: Anonymous (78 y/o, anxiety)
**Discrepancies Detected**: 3 (1 Critical, 1 High, 1 Moderate)
**Generated by**: MedGemma-4B (Simulation Engine)

---

### 🔴 CRITICAL — Financial Abuse / Basic Needs
• **Discrepancy**: Patient anonymously reported losing life savings to a "lottery" phone scam and can no longer afford heart medication. Clinical notes only mention anxiety and high blood pressure.
• **Suggested approach**: *"I'm really concerned that you're skipping your heart medicine. Sometimes things happen with our finances that make it hard to afford these things. Are you dealing with any unexpected financial stress or has anyone been pressuring you for money?"*
• **Clinical reasoning**: Elder financial exploitation is a critical safety issue that directly impacts medical compliance and physical health (heart failure risk).

---
**Summary**: 3 discrepancies. Financial fraud is preventing medication adherence. **Urgent social work referral for financial protection and medication assistance programs required.**""",

        "sexual_health": """## 🚨 TruthShield Clinical Alert — HIGH
**Patient**: Anonymous (24 y/o, dermatology)
**Discrepancies Detected**: 2 (0 Critical, 2 High)
**Generated by**: MedGemma-4B (Simulation Engine)

---

### 🟡 HIGH — Sexual Health / Concealed STI
• **Discrepancy**: Patient anonymously reported fear of an STI from a new partner. Clinician was told it was an "allergic reaction to soap." 
• **Suggested approach**: *"I know we talked about this being a soap allergy, but I also see this a lot when people have been exposed to something new sexually. This is a judgment-free space—is there any chance these bumps could be related to a recent partner?"*
• **Clinical reasoning**: Sensitive screening for STIs is often masked by patient embarrassment. Left untreated, certain infections can lead to long-term reproductive health issues.

---
        ---
---
**Summary**: 2 discrepancies. Patient is masking STI concerns due to stigma. **Switch to full STI screening panel and provide confidential counseling.**""",
        
        "general": """## 🚨 TruthShield Clinical Alert — MODERATE
**Patient**: Anonymous (Diagnostic Check-in)
**Discrepancies Detected**: 1 (0 Critical, 0 High, 1 Moderate)
**Generated by**: MedGemma-4B (Simulation Engine)

---

### 🟢 MODERATE — Generalized Disclosure Discrepancy
• **Discrepancy**: The patient's structured responses suggest clinical symptoms (e.g., fatigue, anxiety) that were not fully disclosed during the initial verbal interview or documented in the primary notes.
• **Suggested approach**: *"I've noticed some things in the survey that we didn't get a chance to talk about yet. It seems like you've been feeling more tired or anxious lately than you mentioned earlier. Can you tell me more about that?"*
• **Clinical reasoning**: Unvoiced symptoms often hide underlying social stressors or mental health concerns that patients are hesitant to bring up directly.

---
**Summary**: 1 discrepancy detected. Minimal clinical misalignment found. **Recommend a brief open-ended follow-up to address unvoiced symptoms.**"""
    }

    return SIMULATED_ALERTS.get(scenario_id, "No simulated alert available for this scenario.")


# ─────────────────────────────────────────────────────────────────────────────
# AI INFERENCE BUILDERS — Formats data for MedGemma
# ─────────────────────────────────────────────────────────────────────────────

def build_full_prompt(survey_responses: str, clinical_notes: str,
                      patient_age: str = "Unknown",
                      visit_type: str = "Routine") -> str:
    """Builds the complete prompt chain for MedGemma inference."""

    user_prompt = DISCREPANCY_ANALYSIS_PROMPT.format(
        survey_responses=survey_responses,
        clinical_notes=clinical_notes
    )

    return user_prompt


def build_alert_prompt(discrepancy_analysis: str,
                       patient_age: str = "Unknown",
                       visit_type: str = "Routine") -> str:
    """Builds the clinician alert prompt from analyzed discrepancies."""

    return CLINICIAN_ALERT_PROMPT.format(
        discrepancy_analysis=discrepancy_analysis,
        patient_age=patient_age,
        visit_type=visit_type
    )
