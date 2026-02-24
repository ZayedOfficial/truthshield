"""
Scenarios Library for TruthShield Demo Mode.
Contains realistic clinical cases focusing on social and emotional trauma with 10-MCQ depth.
"""

SCENARIOS = {
    "cyberbullying": {
        "title": "Adolescent Cyberbullying",
        "department": "Pediatrics & Adolescent Medicine",
        "age": "14",
        "visit_type": "Wellness Check",
        "survey": "I feel like everyone at school hates me. I can't sleep because my phone keeps pinging with mean messages. I don't want to go to school anymore. My stomach hurts every morning.",
        "notes": "Patient presents for annual wellness exam. Mother reports patient has been 'moody' and 'withdrawn'. Physical exam normal, though patient appears tired. Denies SI/HI. Documented as 'typical adolescent adjustment'.",
        "mcqs": [
            "Do you experience severe stomach pain in the morning?",
            "Do you have trouble falling or staying asleep?",
            "Do you often feel sad or down about school?",
            "Have you been withdrawing from your close friends?",
            "Do you suffer from daily headaches?",
            "Have you noticed any big changes in your appetite?",
            "Do you feel anxious when using social media?",
            "Are you trying to avoid going to school lately?",
            "Do you find yourself becoming irritable with your parents?",
            "Do you ever feel hopeless about the situation at school?"
        ],
        "is_social": True
    },
    "caregiver_burnout": {
        "title": "Caregiver Burnout",
        "department": "Psychiatry & Behavioral Health",
        "age": "52",
        "visit_type": "Chronic Pain Follow-up",
        "survey": "I am so tired I can barely think. My mother has dementia and I'm the only one looking after her. I haven't slept more than 4 hours in months. I'm starting to feel resentful and then I feel guilty. My back is killing me and I'm getting migraines.",
        "notes": "Patient presents with escalating chronic low back pain and new-onset migraines. Reports high stress but 'managing'. Physical exam reveals significant muscle tension in trapezius and lumbar region. Documented as 'mechanical back pain'.",
        "mcqs": [
            "Do you feel constantly fatigued during the day?",
            "Is your back pain preventing you from daily tasks?",
            "Are you suffering from daily or frequent headaches?",
            "Do you feel overwhelmed by your caregiving duties?",
            "Do you often feel guilty about your own feelings?",
            "Have you lost interest in activities you once enjoyed?",
            "Do you feel significant muscle tension in your neck/back?",
            "Do you find it difficult to concentrate on tasks?",
            "Do you feel socially isolated from friends or family?",
            "Do you feel anxious when thinking about the future?"
        ],
        "is_social": True
    },
    "hidden_grief": {
        "title": "Concealed Grief",
        "department": "Geriatrics / Internal Medicine",
        "age": "65",
        "visit_type": "Post-CVA Follow-up",
        "survey": "Everything feels empty. My partner passed away six months ago and I haven't told the doctor because I don't want to seem weak or like I can't live alone. I've stopped taking my blood pressure meds because I just don't care anymore.",
        "notes": "Patient is 1-year post-CVA. Blood pressure is elevated (155/95). Patient claims to be taking all medications as prescribed. Mood appears stable, 'stoic'. Pharmacy records indicate possible gap in refills.",
        "mcqs": [
            "Have you lost interest in your regular activities?",
            "Are you sometimes skipping your prescribed medications?",
            "Do you feel your blood pressure is higher than usual?",
            "Do you often feel lonely or isolated?",
            "Do you feel a sense of emptiness in your daily life?",
            "Are you having significant trouble with sleep?",
            "Have you noticed a persistent loss of appetite?",
            "Are you withdrawing from social interactions?",
            "Do you feel hopeless about your current health?",
            "Do you experience constant fatigue or low energy?"
        ],
        "is_social": True
    },
    "domestic_violence": {
        "title": "Physical Safety (Restricted)",
        "department": "Emergency Medicine / Social Work",
        "age": "28",
        "visit_type": "Injury Evaluation",
        "survey": "I didn't fall. He pushed me because I didn't have dinner ready. I'm terrified to go home but I have nowhere else to go. Please don't tell him I said this.",
        "notes": "Patient presents with bruising on left arm and upper ribs. Explains injuries as 'tripping over the dog'. Partner is present in the room and very attentive. Patient is quiet and avoids eye contact.",
        "mcqs": [
            "Are your current injuries causing severe pain?",
            "Do you feel afraid of your partner currently?",
            "Does your partner restrict where you can go?",
            "Does your partner try to control who you see?",
            "Are you experiencing significant sleep disturbance?",
            "Do you feel a constant sense of anxiety?",
            "Do you feel like you have to be 'on guard' at home?",
            "Have you become isolated from your family/friends?",
            "Do you feel helpless in your current situation?",
            "Have you had frequent 'accidents' or injuries lately?"
        ],
        "is_social": True
    },
    "veteran_trauma": {
        "title": "Veteran Re-integration",
        "department": "Psychiatry / VA Care",
        "age": "34",
        "visit_type": "Routine Check-up",
        "survey": "I can't handle the noise at the grocery store. I feel like I'm always looking for exits. I'm drinking much more than I tell the nurse just to stop the dreams.",
        "notes": "Veteran patient for annual physical. Stoic presentation. Denies PTSD symptoms on standardized screen (PC-PTSD: 0). Reports 'social drinking' only. Documented as 'stable post-service transition'.",
        "mcqs": [
            "Are you experiencing frequent nightmares?",
            "Do you feel hyper-vigilant in public spaces?",
            "Has your alcohol consumption increased recently?",
            "Do you find yourself avoiding certain places or people?",
            "Are you experiencing intrusive 'flashbacks'?",
            "Do you feel more irritable or angry than usual?",
            "Are sleep issues impacting your daily life?",
            "Do you feel extreme anxiety in crowded environments?",
            "Do you feel emotionally 'numb' or disconnected?",
            "Do you have a strong startle response to loud noises?"
        ],
        "is_social": True
    },
    "financial_fraud": {
        "title": "Elder Financial Fraud",
        "department": "Geriatrics",
        "age": "78",
        "visit_type": "Anxiety Follow-up",
        "survey": "I'm so ashamed. I gave my bank details to someone on the phone who said they were from the lottery. Now my savings are gone and I can't pay for my heart medication this month. I'm terrified they'll come to my house.",
        "notes": "Patient presents with generalized anxiety and palpitations. Vitals stable. Claims to be 'finishing old bottle' of medication before picking up refill. Living alone since spouse passed.",
        "mcqs": [
            "Do you feel severe anxiety about your finances?",
            "Are you experiencing regular heart palpitations?",
            "Is financial distress impacting your medical care?",
            "Are you skipping medications to save money?",
            "Do you feel unsafe in your own home?",
            "Do you feel a deep sense of shame or embarrassment?",
            "Is insomnia preventing you from resting?",
            "Have you noticed a loss of appetite lately?",
            "Are you withdrawing from social activities?",
            "Are you experiencing sudden panic or terror?"
        ],
        "is_social": True
    },
    "sexual_health": {
        "title": "Sensitive Sexual Health",
        "department": "Primary Care / Sexual Health",
        "age": "24",
        "visit_type": "Dermatology Intake",
        "survey": "I have these bumps but I was too embarrassed to tell the doctor it might be from a new partner. I said it was just a rash from new soap. I'm really scared it's something permanent.",
        "notes": "Patient presents with 'localized allergic reaction' on thigh/groin area. Denies new sexual partners. Requests prescription-strength hydrocortisone. Appeared anxious during physical exam.",
        "mcqs": [
            "Are the lesions or bumps causing you distress?",
            "Do you feel a high level of anxiety about your health?",
            "Are you afraid of a specific medical diagnosis?",
            "Do you feel embarrassed to discuss your symptoms?",
            "Are you concerned about a recent sexual partner?",
            "Does fear of 'stigma' prevent you from speaking up?",
            "Do you find it hard to trust the medical team?",
            "Are you experiencing pain or itching in the area?",
            "Is worry about this condition causing sleep loss?",
            "Would you prefer a confidential STI screening panel?"
        ],
        "is_social": True
    }
}

def get_scenario_list():
    return [f"{v['title']} ({k})" for k, v in SCENARIOS.items()]

def get_scenario(id_or_title):
    if "(" in id_or_title:
        id_key = id_or_title.split("(")[-1].replace(")", "").strip()
        return SCENARIOS.get(id_key)
    return SCENARIOS.get(id_or_title)
