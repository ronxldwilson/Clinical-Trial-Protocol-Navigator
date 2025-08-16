# Clinical-Trial-Protocol-Navigator


# 1) Normalize your trial JSON

### 1.1 Fields to keep (flatten)

* IDs & links: `TrialID`, `web_address`
* Titles: `Public_title`, `Scientific_title`
* Status/meta: `Recruitment_Status`, `Study_type`, `Study_design`, `Phase`, `Target_size`, `Countries`
* Criteria: `Inclusion_Criteria`, `Exclusion_Criteria` (may be merged into one field in some registries)
* Condition(s): `Condition`
* Dates (optional): `Date_registration`, `Last_Refreshed_on`

### 1.2 Cleaning tips

* Strip HTML (`<br>`, `\r\n`) and bullets, normalize whitespace.
* Standardize age: map `"5 Years"` → `5`, `"N/A"` → `None`.
* Split multi-values: e.g., `Condition` split by `;`, `Intervention` split by `;`.
* Keep **obesity-only** trials. Filter where `Condition` or `Public_title`/`Scientific_title` match: `obesity|overweight|BMI|weight management|adiposity|metabolic syndrome`.

---

# 2) Extract machine-readable rules (light IE)

You don’t need a full rule engine; just grab the **common patterns** seen in obesity trials:

* **Age range**: `(\d{1,3})\s*(?:to|-|–)\s*(\d{1,3})\s*years|≥?\s*(\d{1,3})\s*years`
* **BMI thresholds**: `BMI\s*(≥|>|≤|<|between)\s*(\d{1,2}\.?\d*)` (often “BMI ≥ 30” or “BMI 27–40”)
* **Comorbidities allowed/excluded**: look for diabetes (T1/T2), uncontrolled HTN, pregnancy/lactation, psychiatric illness, recent surgery, medications (GLP-1, tirzepatide), bariatric surgery history.
* **Payer/coverage constraints** (like your sample with Medicaid/MO HealthNet)—keep as a **site/insurance filter**.
* **Country/site**: for location matching.

Store into a normalized object per trial:

```json
{
  "trial_id": "NCT07049861",
  "url": "https://clinicaltrials.gov/ct2/show/NCT07049861",
  "status": "Recruiting",
  "countries": ["United States"],
  "age_min": 5,
  "age_max": null,
  "bmi_min": 30.0,
  "bmi_max": null,
  "requires_medicaid": true,
  "exclusions": ["pregnancy", "no obesity", "not at participating clinic"],
  "raw_inclusion": "...cleaned text...",
  "raw_exclusion": "...cleaned text..."
}
```

> Don’t overfit: if a trial doesn’t mention BMI at all, leave it `null` and let the model decide from text during inference.

---

# 3) Define patient profile schema (synthetic)

For obesity, a compact schema that covers 90% of eligibility checks:

```json
{
  "age": 42,
  "sex": "Female",
  "country": "United States",
  "bmi": 34.2,
  "diagnoses": ["obesity", "hypertension"],
  "meds": ["metformin"],               // include GLP-1s/tirzepatide if relevant
  "pregnant": false,
  "bariatric_surgery_history": false,
  "insurance": "Medicaid",             // or null
  "site_affiliation": "FQHC-Missouri", // optional: clinic/site names
  "preferences": { "max_distance_km": 200 }
}
```

Generate **positives** (meets all clear rules) and **hard negatives** (violates exactly one salient rule: BMI 29.5, age below min, not Medicaid, pregnant true, etc.).

---

# 4) Build training pairs (instruction SFT)

Two strong formats—pick one (or do both):

### A) **Per-trial eligibility** (simpler, very effective)

* **Input**: patient profile + one trial (title + cleaned inclusion/exclusion + country + status)
* **Output** (JSON): `eligible` (true/false), `met_criteria`, `failed_criteria`, `rationale`, `confidence`.

**Training sample (JSONL record):**

```json
{
  "instruction": "Decide eligibility for the patient against the trial and explain.",
  "input": {
    "patient": {...},
    "trial": {...}
  },
  "output": {
    "eligible": true,
    "met_criteria": ["Age ≥ 5", "BMI ≥ 30", "Medicaid recipient"],
    "failed_criteria": [],
    "rationale": "Patient meets all inclusion and no exclusion appears triggered.",
    "confidence": 0.88
  }
}
```

### B) **Top-K ranking** (shows “matching” ability)

* Retrieve K candidate trials for a patient (e.g., K=5) with keywords/embeddings.
* **Input**: patient + list of K trial snippets.
* **Output**: ranked list with eligibility + reason per trial.

---

# 5) Dataset sizing & splits

* With 723 trials, create \~**10–20 pairs per trial** (5–10 positives, 5–10 hard negatives) ⇒ **7k–14k** samples.
* **Split by trial** (not by pair) into train/val/test = 80/10/10 to prevent leakage.
* Keep a tiny **human-reviewed test set** (e.g., 150–300 pairs) to showcase accuracy and error types.

---

# 6) Fine-tuning setup (gpt-oss-20b first)

Use **QLoRA** (PEFT) for speed/VRAM economy.

### 6.1 File layout

```
data/
  train.jsonl
  val.jsonl
  test.jsonl
scripts/
  prepare_data.py
  finetune_lora.py
```

### 6.2 JSONL format (one example per line)

For **Per-trial eligibility**:

```json
{"instruction":"Decide eligibility...", "input":{"patient":{...}, "trial":{...}}, "output":{...}}
```

### 6.3 Training hyperparams (starting point)

* Base: `gpt-oss-20b` (BF16 if possible; else 4-bit + QLoRA)
* LoRA: `r=16, alpha=32, dropout=0.05`, target modules: attention + MLP
* Seq len: 4k tokens (eligibility text can be long)
* Batch size (effective): 64 (gradient\_accumulation to fit GPU)
* LR: 2e-4 (LoRA), cosine decay, 1–2 epochs over \~10k–20k samples
* Loss: next-token LM; **format the conversation** as system/instruction/input/output to stabilize.

*(If you want, I can give you a ready-to-run `transformers + peft` training script.)*

---

# 7) Evaluation

### 7.1 Automatic

* On **per-trial** test set:

  * Eligibility **precision/recall/F1**
  * Error breakdown: “missed exclusion” vs “false inclusion”
* On **ranking** test set:

  * **MRR\@K / nDCG\@K**, plus “% of true-eligible trials in top-3”

### 7.2 Human spot-check

* 50 random test examples: is rationale correct? are criteria accurate? any hallucinated rules?

---

# 8) Inference pipeline for the demo

You have two clean paths; both are hackathon-friendly.

**Option A (closed corpus, fastest):**

* Preselect \~200 high-quality, **Recruiting** obesity trials.
* Demo flow:

  1. Enter patient profile in the UI.
  2. Retrieve top-K trials with a **keyword index** (BM25) over `Condition`, `Inclusion_Criteria`, country.
  3. For each retrieved trial, call your **fine-tuned model** with **Per-trial eligibility** prompt.
  4. Sort by `eligible=true` then by model confidence.
  5. Render cards: title, site/country, “why eligible”, link.

**Option B (hybrid retrieval, scalable):**

* Put all 723 trials into a small **SQLite** (or vector DB).
* Use simple **embedding** retrieval (e.g., MiniLM) to get top-K, then the same model pass as above.
* Same UI; now it generalizes to trials **unseen** in training (good for judges).

---

# 9) Prompt templates (stable formatting)

**System:**

```
You are a clinical trial eligibility assistant. Be precise and conservative. Do not infer missing facts.
Return only valid JSON exactly matching the schema.
```

**User (Per-trial):**

```
PATIENT:
{patient_json}

TRIAL:
{trial_json}

TASK:
1) Decide eligibility (true/false).
2) List met_criteria and failed_criteria using phrases present in the trial text.
3) Give a brief rationale.
4) Provide a confidence in [0,1].

Return JSON with keys: eligible, met_criteria, failed_criteria, rationale, confidence.
```

This same format is used during training (as instruction SFT), so inference is stable.

---

# 10) Demo polish (what judges notice)

* **Two modes**: “Patient-friendly” summary toggle → render a lay summary *from the same JSON* (no extra model calls).
* **Safety banner**: “This tool is assistive. Not medical advice. Clinician review required.”
* **Latency**: Cap K=5–8 trials per request; cache trial embeddings; stream model output.
* **Offline story**: mention it runs locally with vLLM/Ollama + SQLite (even if you host the demo).

---

# 11) Milestone plan (7–10 working days)

**Day 1–2**

* Clean JSON, filter obesity, parse rough rules (age/BMI/country/clear exclusions).
* Generate 3–5 synthetic patients per trial (positives + hard negatives).

**Day 3–4**

* Build 7k–10k pairs; split by trial; baseline eval with a simple rules engine (sanity check).
* Prepare JSONL for SFT.

**Day 5–6**

* LoRA fine-tune on gpt-oss-20b; iterate 1–2 times; pick best checkpoint.
* Write evaluator script (precision/recall; error types).

**Day 7–8**

* Build minimal UI + retrieval (BM25 or small embeddings).
* Wire model inference; render ranked cards + explanations.

**Day 9–10**

* Human review set; fix common error modes; record 3-min demo video.

---

