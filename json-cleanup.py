import json
import re
import sys

def normalize_trial(trial):
    """Normalize a single trial entry into a consistent structure"""
    normalized = {
        "trial_id": trial.get("TrialID"),
        "url": trial.get("web_address"),
        "status": trial.get("Recruitment_Status"),
        "countries": [c.strip() for c in trial.get("Countries", "").split(";") if c.strip()],
        "age_min": None,
        "age_max": None,
        "bmi_min": None,
        "bmi_max": None,
        "requires_medicaid": False,
        "exclusions": [],
        "raw_inclusion": trial.get("Inclusion_Criteria", ""),
        "raw_exclusion": trial.get("Exclusion_Criteria", "")
    }

    # Age extraction
    if trial.get("Inclusion_agemin") and trial["Inclusion_agemin"].isdigit():
        normalized["age_min"] = int(trial["Inclusion_agemin"])
    if trial.get("Inclusion_agemax") and trial["Inclusion_agemax"].isdigit():
        normalized["age_max"] = int(trial["Inclusion_agemax"])

        # BMI extraction from inclusion criteria text
    inclusion_text = trial.get("Inclusion_Criteria", "").lower()
    bmi_match = re.search(
        r"bmi\s*(≥|>|≤|<|between)?\s*([\d\.]+)(?:\s*[-to]+\s*([\d\.]+))?",
        inclusion_text
    )
    if bmi_match:
        def safe_float(val):
            if not val:
                return None
            # remove trailing dots, commas, semicolons, etc.
            cleaned = val.strip().rstrip(".,;:")
            try:
                return float(cleaned)
            except ValueError:
                return None

        if bmi_match.group(2):
            normalized["bmi_min"] = safe_float(bmi_match.group(2))
        if bmi_match.group(3):
            normalized["bmi_max"] = safe_float(bmi_match.group(3))


    # Medicaid / payer requirement
    if "medicaid" in inclusion_text or "mo healthnet" in inclusion_text:
        normalized["requires_medicaid"] = True

    # Common exclusions
    exclusion_text = trial.get("Exclusion_Criteria", "").lower()
    for keyword in ["pregnancy", "no obesity", "not at participating clinic",
                    "hypertension", "psychiatric", "surgery"]:
        if keyword in exclusion_text:
            normalized["exclusions"].append(keyword)

    return normalized


# ---- Run on your JSON ----
with open("obesity.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Debug: show top-level structure
print("Top-level JSON type:", type(data))
if isinstance(data, dict):
    print("Keys in JSON:", list(data.keys())[:10])  # show first 10 keys

# ✅ Handle {"Trials": [...]} OR a list
if isinstance(data, dict) and "Trial" in data:
    trials = data["Trial"]
elif isinstance(data, list):
    trials = data
else:
    print("❌ ERROR: JSON does not contain a list of trials at top level.")
    sys.exit(1)

print(f"Found {len(trials)} trials")

normalized_trials = [normalize_trial(trial) for trial in trials]

# Save normalized JSON
with open("obesity_normalized.json", "w", encoding="utf-8") as f:
    json.dump(normalized_trials, f, indent=2, ensure_ascii=False)

print(f"✅ Converted {len(normalized_trials)} trials into normalized format")
