"""Generate synthetic, pseudonymous patient records into data/records/.

- No names, addresses, phone numbers, emails, or dates of birth. Pseudonymous IDs (P-0001...),
  age bands only.
- Deterministic: seeded randomness (Faker.seed(42) + random.Random(42)), so output is reproducible.
- Template-based (no LLM, no cost).

Usage:
    uv run python scripts/generate_records.py [--count 28] [--out data/records]
"""

import argparse
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from faker import Faker


@dataclass(frozen=True)
class Profile:
    condition: str
    complaint: str
    hpi: str
    medications: list[str]
    labs: list[str]
    assessment: str
    plan: str


PROFILES: list[Profile] = [
    Profile(
        "Essential hypertension",
        "Recurring headaches for {weeks} weeks.",
        "Dull occipital headaches, worse in the morning. Home blood pressure readings have "
        "been consistently elevated. No vision changes or chest pain.",
        ["Lisinopril 10 mg once daily"],
        [
            "Blood pressure 152/96 mmHg (repeat 148/94 mmHg)",
            "Creatinine 0.9 mg/dL",
            "Potassium 4.3 mmol/L",
        ],
        "Essential hypertension, not at goal on current therapy.",
        "Increase lisinopril to 20 mg daily. Keep a home blood pressure log. Reduce salt intake. "
        "Recheck in 4 weeks.",
    ),
    Profile(
        "Type 2 diabetes mellitus",
        "Increased thirst and frequent urination for {weeks} weeks.",
        "Reports polyuria, polydipsia, and blurred vision in the evenings. Diet has included "
        "frequent sugary drinks. No numbness in the feet.",
        ["Metformin 1000 mg twice daily"],
        ["HbA1c 8.4%", "Fasting glucose 182 mg/dL", "Urine albumin-creatinine ratio 25 mg/g"],
        "Type 2 diabetes mellitus with suboptimal glycemic control.",
        "Add empagliflozin 10 mg daily. Refer to dietitian for diabetes education. Annual eye "
        "exam. Recheck HbA1c in 3 months.",
    ),
    Profile(
        "Moderate persistent asthma",
        "Wheezing and nighttime cough for {weeks} weeks.",
        "Symptoms worsen with cold air and exercise. Uses rescue inhaler about four times per "
        "week and wakes at night twice weekly.",
        ["Albuterol inhaler 2 puffs as needed", "Fluticasone 110 mcg 2 puffs twice daily"],
        ["Peak expiratory flow 68% of predicted", "Blood eosinophils 450 cells/uL"],
        "Moderate persistent asthma, not well controlled.",
        "Step up to fluticasone-salmeterol combination inhaler. Review spacer technique. "
        "Written asthma action plan provided.",
    ),
    Profile(
        "Primary hypothyroidism",
        "Fatigue, weight gain, and cold intolerance for {weeks} weeks.",
        "Gained about 4 kg despite no change in diet. Reports dry skin and constipation. "
        "Adherent to thyroid medication.",
        ["Levothyroxine 75 mcg once daily before breakfast"],
        ["TSH 9.8 mIU/L", "Free T4 0.7 ng/dL"],
        "Primary hypothyroidism, under-replaced.",
        "Increase levothyroxine to 88 mcg daily. Recheck TSH in 6 weeks.",
    ),
    Profile(
        "Gastroesophageal reflux disease",
        "Burning chest discomfort after meals for {weeks} weeks.",
        "Heartburn after large or late meals, worse when lying down. Occasional sour taste in "
        "the mouth. No difficulty swallowing and no weight loss.",
        ["Omeprazole 20 mg once daily before breakfast"],
        ["Helicobacter pylori breath test negative", "Hemoglobin 13.9 g/dL"],
        "Gastroesophageal reflux disease without alarm features.",
        "Continue omeprazole for 8 weeks. Avoid late meals, raise head of bed. Endoscopy if "
        "symptoms persist.",
    ),
    Profile(
        "Migraine with aura",
        "Throbbing one-sided headaches with visual aura for {weeks} weeks.",
        "Zigzag lights precede headaches by 20 minutes. Attacks last up to a day with nausea "
        "and light sensitivity, about five per month.",
        ["Sumatriptan 50 mg as needed at headache onset"],
        ["Complete blood count normal", "MRI brain unremarkable"],
        "Migraine with aura, frequent episodic.",
        "Start propranolol 40 mg daily for prevention. Keep a headache diary. Limit triptan use "
        "to fewer than 10 days per month.",
    ),
    Profile(
        "Hyperlipidemia",
        "Routine follow-up for high cholesterol.",
        "No chest pain or claudication. Diet improved modestly. Taking statin most days.",
        ["Atorvastatin 40 mg once daily"],
        [
            "LDL cholesterol 168 mg/dL",
            "Total cholesterol 251 mg/dL",
            "HDL 38 mg/dL",
            "Triglycerides 210 mg/dL",
        ],
        "Hyperlipidemia above LDL target.",
        "Increase atorvastatin to 80 mg daily. Recheck lipid panel in 12 weeks. Encourage "
        "aerobic exercise.",
    ),
    Profile(
        "Atrial fibrillation",
        "Palpitations and lightheadedness for {weeks} weeks.",
        "Episodes of a racing, irregular heartbeat lasting hours. No fainting. Mild shortness "
        "of breath when climbing stairs.",
        ["Apixaban 5 mg twice daily", "Metoprolol succinate 50 mg once daily"],
        [
            "ECG: irregularly irregular rhythm, ventricular rate 112 bpm",
            "TSH 1.9 mIU/L",
            "Echocardiogram: ejection fraction 55%",
        ],
        "Atrial fibrillation with rapid ventricular response. CHA2DS2-VASc score 3.",
        "Continue anticoagulation with apixaban. Increase metoprolol for rate control. "
        "Cardiology referral.",
    ),
    Profile(
        "Chronic obstructive pulmonary disease",
        "Progressive shortness of breath over {weeks} weeks.",
        "Breathless after walking one block. Morning cough with white sputum. Former smoker, "
        "30 pack-years, quit three years ago.",
        ["Tiotropium 18 mcg inhaled once daily", "Albuterol inhaler as needed"],
        ["Spirometry FEV1/FVC 0.62, FEV1 54% of predicted", "Oxygen saturation 93% on room air"],
        "Chronic obstructive pulmonary disease, moderate (GOLD 2).",
        "Refer to pulmonary rehabilitation. Influenza and pneumococcal vaccines. Add "
        "long-acting beta agonist if no improvement.",
    ),
    Profile(
        "Chronic kidney disease stage 3b",
        "Ankle swelling for {weeks} weeks.",
        "Bilateral ankle edema by evening. Urine looks foamy. Long history of high blood "
        "pressure. Uses over-the-counter pain relievers occasionally.",
        ["Losartan 50 mg once daily", "Furosemide 20 mg once daily"],
        [
            "eGFR 41 mL/min/1.73m2",
            "Creatinine 1.7 mg/dL",
            "Urine albumin-creatinine ratio 210 mg/g",
        ],
        "Chronic kidney disease stage 3b with albuminuria.",
        "Avoid NSAIDs. Nephrology referral. Recheck kidney function and potassium in 2 weeks.",
    ),
    Profile(
        "Iron deficiency anemia",
        "Fatigue and shortness of breath on exertion for {weeks} weeks.",
        "Feels tired climbing stairs. Craves ice. Reports heavy menstrual periods. No black "
        "stools.",
        ["Ferrous sulfate 325 mg once daily"],
        ["Hemoglobin 9.1 g/dL", "Ferritin 6 ng/mL", "MCV 71 fL"],
        "Iron deficiency anemia, likely from menstrual blood loss.",
        "Continue oral iron with vitamin C. Gynecology referral. Repeat blood count in 6 weeks.",
    ),
    Profile(
        "Major depressive disorder",
        "Low mood and poor sleep for {weeks} weeks.",
        "Loss of interest in hobbies, early morning waking, and low energy. Denies thoughts of "
        "self-harm. Good family support.",
        ["Sertraline 50 mg once daily"],
        ["PHQ-9 score 17", "TSH 2.1 mIU/L"],
        "Major depressive disorder, moderate, single episode.",
        "Continue sertraline. Refer for cognitive behavioral therapy. Follow up in 2 weeks.",
    ),
    Profile(
        "Osteoarthritis of the knee",
        "Right knee pain for {weeks} weeks, worse on stairs.",
        "Morning stiffness lasting under 30 minutes. Pain after long walks. No locking or "
        "giving way.",
        ["Acetaminophen 1 g three times daily", "Topical diclofenac gel to right knee"],
        ["Knee X-ray: medial joint space narrowing with osteophytes", "ESR 12 mm/h"],
        "Osteoarthritis of the right knee.",
        "Physical therapy for quadriceps strengthening. Weight management. Consider steroid "
        "injection if no improvement.",
    ),
    Profile(
        "Urinary tract infection",
        "Painful urination and urinary frequency for {days} days.",
        "Burning on urination and urgency. No fever or flank pain. Similar episode last year.",
        ["Nitrofurantoin 100 mg twice daily for 5 days"],
        [
            "Urinalysis positive for nitrites and leukocyte esterase",
            "Urine culture: Escherichia coli, sensitive to nitrofurantoin",
        ],
        "Uncomplicated lower urinary tract infection.",
        "Complete nitrofurantoin course. Increase fluid intake. Return if fever or back pain.",
    ),
    Profile(
        "Gout",
        "Sudden painful swelling of the left big toe for {days} days.",
        "Pain began overnight after a large meal. The joint is red, hot, and too painful to "
        "touch. One similar attack two years ago.",
        ["Colchicine 0.6 mg twice daily", "Allopurinol 100 mg once daily (start after flare)"],
        ["Serum uric acid 9.2 mg/dL", "Joint aspirate: needle-shaped urate crystals"],
        "Acute gout flare of the first metatarsophalangeal joint.",
        "Colchicine for the flare. Start allopurinol after resolution with uric acid target "
        "below 6 mg/dL. Limit alcohol and red meat.",
    ),
    Profile(
        "Osteoporosis",
        "Back pain after a minor fall {weeks} weeks ago.",
        "Slipped at home and landed on her back. X-ray showed a mild vertebral compression "
        "fracture. Height loss of 3 cm over five years.",
        ["Alendronate 70 mg once weekly", "Calcium carbonate 600 mg with vitamin D3 800 IU daily"],
        ["DEXA T-score -2.8 at the lumbar spine", "Vitamin D 18 ng/mL"],
        "Osteoporosis with vertebral compression fracture.",
        "Start alendronate and vitamin D. Fall-prevention review. Repeat DEXA in 2 years.",
    ),
]

SECONDARY_HISTORY = [
    "Seasonal allergic rhinitis.",
    "Appendectomy in childhood.",
    "Tonsillectomy at age 12.",
    "Carpal tunnel release, left wrist.",
    "Cholecystectomy five years ago.",
    "Childhood eczema, resolved.",
    "Fractured right wrist ten years ago.",
    "No prior surgeries.",
]

# Rare allergies appear exactly once so golden questions have a single correct patient.
RARE_ALLERGIES = [
    "Sulfa drugs (hives).",
    "Latex (contact dermatitis).",
    "Codeine (severe nausea and vomiting).",
    "Shellfish (anaphylaxis). Carries an epinephrine auto-injector.",
    "Peanuts (throat swelling).",
    "Aspirin (bronchospasm).",
    "Iodinated contrast dye (hives).",
    "Bee stings (anaphylaxis).",
    "Erythromycin (severe diarrhea).",
]
COMMON_ALLERGIES = ["Penicillin (rash).", "Penicillin (rash).", "Penicillin (hives)."]

SEXES = ["F", "M"]
AGE_BANDS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]


def build_record(
    patient_id: str, profile: Profile, allergy: str, rng: random.Random, record_date: date
) -> str:
    sex = "F" if "menstrual" in profile.hpi or "her back" in profile.hpi else rng.choice(SEXES)
    age_band = rng.choice(AGE_BANDS)
    complaint = profile.complaint.format(weeks=rng.randint(2, 8), days=rng.randint(2, 5))
    history = [
        f"{profile.condition} diagnosed {rng.randint(2012, 2023)}.",
        rng.choice(SECONDARY_HISTORY),
    ]
    return "\n".join(
        [
            f"# Patient Record: {patient_id}",
            f"- Age band: {age_band}",
            f"- Sex: {sex}",
            f"- Record date: {record_date.isoformat()}",
            "",
            "## Chief Complaint",
            complaint,
            "",
            "## History of Present Illness",
            profile.hpi,
            "",
            "## Past Medical History",
            " ".join(history),
            "",
            "## Medications",
            "\n".join(f"- {m}" for m in profile.medications),
            "",
            "## Allergies",
            allergy,
            "",
            "## Lab Results",
            "\n".join(f"- {lab}" for lab in profile.labs),
            "",
            "## Assessment",
            profile.assessment,
            "",
            "## Plan",
            profile.plan,
            "",
        ]
    )


def generate(count: int, out_dir: Path) -> list[Path]:
    Faker.seed(42)
    fake = Faker()
    rng = random.Random(42)

    # Every profile at least once, then fill with repeats.
    profiles = PROFILES + [rng.choice(PROFILES) for _ in range(max(0, count - len(PROFILES)))]
    rng.shuffle(profiles)
    profiles = profiles[:count]

    allergies = RARE_ALLERGIES + COMMON_ALLERGIES
    allergies += ["No known drug allergies."] * max(0, count - len(allergies))
    rng.shuffle(allergies)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (profile, allergy) in enumerate(zip(profiles, allergies, strict=False), start=1):
        patient_id = f"P-{i:04d}"
        record_date = fake.date_between(start_date=date(2024, 1, 1), end_date=date(2025, 12, 31))
        path = out_dir / f"{patient_id}.md"
        path.write_text(
            build_record(patient_id, profile, allergy, rng, record_date),
            encoding="utf-8",
            newline="\n",
        )
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=28)
    parser.add_argument("--out", type=Path, default=Path("data/records"))
    args = parser.parse_args()
    files = generate(args.count, args.out)
    print(f"Wrote {len(files)} records to {args.out}")


if __name__ == "__main__":
    main()
