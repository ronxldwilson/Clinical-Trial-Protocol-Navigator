import json
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def safe_num(value, default=0):
    """Convert None or missing values to default."""
    if value is None:
        return default
    return value


def check_eligibility(patient, trial):
    """Return eligibility decision with rationale."""
    age_min = safe_num(trial.get("age_min", 0), 0)
    age_max = safe_num(trial.get("age_max", 120), 120)
    bmi_min = safe_num(trial.get("bmi_min", 0), 0)
    bmi_max = safe_num(trial.get("bmi_max", 100), 100)

    age = safe_num(patient.get("age"), -1)  # if missing, force fail
    bmi = safe_num(patient.get("bmi"), -1)  # if missing, force fail
    pregnant = patient.get("pregnant", False)

    met, failed = [], []

    # --- Age ---
    if age_min <= age <= age_max:
        met.append(f"Age between {age_min} and {age_max}")
    else:
        failed.append(f"Age not in range {age_min}-{age_max}")

    # --- BMI ---
    if bmi_min <= bmi <= bmi_max:
        met.append(f"BMI between {bmi_min} and {bmi_max}")
    else:
        failed.append(f"BMI not in range {bmi_min}-{bmi_max}")

    # --- Pregnancy ---
    if trial.get("exclude_pregnant"):
        if not pregnant:
            met.append("Not pregnant")
        else:
            failed.append("Pregnancy exclusion")

    eligible = len(failed) == 0
    rationale = "Meets all criteria." if eligible else f"Failed criteria: {', '.join(failed)}"

    return {
        "eligible": eligible,
        "met_criteria": met,
        "failed_criteria": failed,
        "rationale": rationale,
        "confidence": 0.8 if eligible else 0.6
    }


def build_dataset(trials_file, patients_file, out_prefix, pairs_per_trial=15):
    with open(trials_file, encoding="utf-8") as f:
        trials = json.load(f)
    with open(patients_file, encoding="utf-8") as f:
        patients = json.load(f)

    records = []
    rng = random.Random(42)

    with tqdm(total=len(trials), desc="Building dataset", unit="trial") as pbar:
        for trial in trials:
            # Shuffle patients each trial for variety
            sampled = rng.sample(patients, min(len(patients), pairs_per_trial * 5))

            positives, negatives = [], []

            for p in sampled:
                result = check_eligibility(p, trial)
                record = {
                    "instruction": "Decide eligibility for the patient against the trial and explain.",
                    "input": {"patient": p, "trial": trial},
                    "output": result,
                }
                if result["eligible"]:
                    positives.append(record)
                else:
                    negatives.append(record)

            # Balance: pick 5–10 positives, 5–10 negatives
            if positives:
                pos_keep = rng.sample(positives, min(len(positives), rng.randint(5, 10)))
            else:
                pos_keep = []  # handle trials with no eligible patients

            if negatives:
                neg_keep = rng.sample(negatives, min(len(negatives), rng.randint(5, 10)))
            else:
                neg_keep = []  # handle trials with no negatives

            records.extend(pos_keep + neg_keep)

            pbar.update(1)

    print(f"\n✅ Collected {len(records)} total pairs")

    # ---- Split by trial IDs ----
    trial_ids = list(range(len(trials)))
    train_ids, test_ids = train_test_split(trial_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

    split_map = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        with open(f"{out_prefix}_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                trial_idx = trials.index(r["input"]["trial"])
                if trial_idx in split_map[split]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    split_counts[split] += 1

    print("\n📊 Dataset splits:")
    for k, v in split_counts.items():
        print(f"{k}: {v} pairs")


if __name__ == "__main__":
    build_dataset(
        "obesity_normalized.json",
        "synthetic_patients.json",
        "eligibility_dataset",
        pairs_per_trial=15,
    )
