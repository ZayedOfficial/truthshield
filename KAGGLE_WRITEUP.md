# Kaggle Submission: TruthShield Content

### [Kaggle Basic Details]
**Title**: TruthShield: Bridging the "Honesty Gap" with MedGemma-4B
**Subtitle**: A privacy-first, on-device honesty layer for clinicians to detect masked patient trauma and discrepancies.
**Tracks**: MedGemma Impact (Advancing Healthcare with Gemma)

---

### [Project Description]

### Project name 
TruthShield

### Your team 
[Team Member Name] — AI Engineer & Clinical Strategy
[Team Member Name] — UX/UI Specialist

### Problem statement
Clinical research shows that **80.9% of patients** withhold information from their doctors—often critical details regarding social trauma, abuse, or medication non-adherence—due to shame or fear of judgment. This "masking" leads to misdiagnosis, inefficient treatments, and missed opportunities to intervene in life-threatening social situations (e.g., domestic violence or financial fraud).

TruthShield addresses this "Honesty Gap" by providing a calm, anonymous digital space for patients to express their truth before and during a clinical visit, turning unspoken trauma into a clear, actionable clinical signal.

### Overall solution: 
TruthShield utilizes **MedGemma-4B** to create an on-device "Honesty Layer." 
1. **The Honesty Portal**: Analyzes patient narratives at the edge to generate 10 personalized, nuanced clinical questions. This allows patients to disclose truths on their own terms.
2. **The Discrepancy Engine**: Compares the official EHR notes against the patient's portal data. MedGemma-4B identifies "masked truths" and suggests specific, non-confrontational Motivational Interviewing (MI) openers for the clinician.
3. **Impact**: By shifting the discovery of "sensitive" info from a verbal interview to an anonymous portal, TruthShield empowers patients and guides clinicians to high-fidelity care.

### Technical details 
*   **Engine**: MedGemma-4B-Instruct optimized with AWQ quantization for high-speed CPU edge inference.
*   **Privacy**: 100% Offline execution. Patient data never leaves the device, meeting the highest standards of clinical privacy.
*   **Feasibility**: Built on Gradio for tablet-ready deployment. Features a sub-50ms simulation layer for presentation stability and a robust clinical department mapping logic.
*   **Interoperability**: SNOMED-CT coded and HL7 FHIR-bundle compatible for seamless integration with legacy EHR systems.
