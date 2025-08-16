import json
import random
from copy import deepcopy

# Load your normalized trial dataset
with open("obesity_normalized.json", "r", encoding="utf-8") as f:
    trials = json.load(f)

# Pools for diversity
sexes = ["Male", "Female"]
countries = ["United States", "India", "Brazil", "Germany"]
diagnoses_pool = ["obesity", "hypertension", "diabetes", "depression"]
meds_pool = ["metformin", "liraglutide", "tirzepatide", "insulin"]
def generate_positive(trial, pid):
    age_min = trial.get("age_min") or 18
    age_max = trial.get("age_max") or 75
    bmi_min = trial.get("bmi_min") or 30
    bmi_max = trial.get("bmi_max") or 50

    # --- Ensure min <= max ---
    if age_min > age_max:
        age_min, age_max = 18, 75   # reset to safe defaults
    if bmi_min > bmi_max:
        bmi_min, bmi_max = 30, 50

    patient = {
        "id": f"P{pid:05d}",
        "age": random.randint(age_min, min(age_max, 70)),
        "sex": random.choice(sexes),
        "country": trial.get("country", "United States"),
        "bmi": round(random.uniform(bmi_min + 0.5, min(bmi_max, 45)), 1),
        "diagnoses": ["obesity"] + random.sample(diagnoses_pool, k=1),
        "meds": random.sample(meds_pool, k=random.randint(0, 1)),
        "pregnant": False,
        "bariatric_surgery_history": False,
        "insurance": "Medicaid" if "medicaid" in trial.get("inclusion_text", "").lower() else random.choice(["Medicaid","Private",None]),
        "site_affiliation": random.choice(["FQHC-Missouri", "General Hospital", None]),
        "preferences": {"max_distance_km": random.choice([50,100,200,500])}
    }
    return patient



def make_hard_negatives(patient, trial, n=3):
    """Create hard negatives by flipping one key rule."""
    negatives = []

    # Rule 1: BMI too low
    if trial.get("bmi_min"):
        neg = deepcopy(patient)
        neg["id"] += "_BMI"
        neg["bmi"] = trial["bmi_min"] - 0.5
        negatives.append(neg)

    # Rule 2: Age too young
    if trial.get("age_min"):
        neg = deepcopy(patient)
        neg["id"] += "_AGE"
        neg["age"] = trial["age_min"] - 1
        negatives.append(neg)

    # Rule 3: Insurance mismatch
    if "medicaid" in trial.get("inclusion_text", "").lower():
        neg = deepcopy(patient)
        neg["id"] += "_INS"
        neg["insurance"] = "Private"
        negatives.append(neg)

    # Rule 4: Pregnant exclusion
    if "pregnan" in trial.get("exclusion_text", "").lower() and patient["sex"] == "Female":
        neg = deepcopy(patient)
        neg["id"] += "_PREG"
        neg["pregnant"] = True
        negatives.append(neg)

    return negatives[:n]  # limit per trial

# --- MAIN ---
patients = []
pid = 0
for trial in trials:
    for _ in range(5):  # generate 5 positives per trial
        pid += 1
        pos = generate_positive(trial, pid)
        patients.append(pos)
        patients.extend(make_hard_negatives(pos, trial))

with open("synthetic_patients.json", "w", encoding="utf-8") as f:
    json.dump(patients, f, indent=2)
