# AI Data Center Database (R1) — Basic Information

## 1. Overview & Scope
This document describes the basic information, field semantics, usability boundaries, and audit risks of `data/AI_database_chrome__llm_final_R1.csv`. It is a data card / schema notes document, not a modeling pipeline specification.

This database is positioned as an **AI-labeled data centers database** built from public-evidence collection followed by LLM judgment.  
The research target is the identification of an **AI-oriented / accelerator-dense / high power-density compatible** infrastructure orientation (an infra-only observable target).  
This phrase refers to infrastructure characteristics that, based only on observable infrastructure fields, are more likely to support dense accelerator-compute deployment. It is not equivalent to AI-load share or utilization.

## 2. R1 Snapshot and Pre-analysis Checklist
### 2.1 R1 Snapshot (based on the current CSV)
1. Number of records: 414 (excluding the header).
2. Number of fields: 24.
3. Noise is clearly present: label errors, marketing-language contamination, and inconsistent evidence granularity.
4. Structural missingness is clearly present: cross-source and cross-company disclosure differences cause heavy concentration of missing values in some fields.

### 2.2 Field Availability Tiers (for pre-modeling prioritization)

| Tier | Field | Non-missing Rate | Role |
|---|---|---|---|
| **T1** (>=70%) | `power_mw` | 90.8% | Core interaction backbone |
| | `cooling` | 79.2% | Core interaction backbone |
| **T2** (40-70%) | `liquid_cool` | 65.0% | Core interaction backbone |
| | `building_sqm` | 59.9% | Core interaction backbone |
| | `rack_kw_typical` | 54.3% | Core interaction backbone |
| | `pue` | 46.6% | Core interaction backbone (borderline) |
| **T3** (<40%) | `whitespace_sqm` | 40.3% | Low coverage, triage only |
| | `rack_kw_peak` | 21.0% | Low coverage, triage only |
| | `rack_density_area_w_per_sf_dc` | 15.5% | Low coverage, triage only |
| | `rack_count` | 7.0% | Extremely low coverage, reference only |

Note: `whitespace_sqm` is near the 40% boundary; by default it is more robust to treat it as a low-coverage field.

### 2.3 Non-random Missingness
Missingness comes from systematic disclosure differences (company strategy, source type, lifecycle stage, and granularity), and **is not random missingness**. The missingness pattern itself may become a model shortcut.

### 2.4 Risk of Mixed Definitions
`IT load vs facility load`, `planned vs operational`, and `campus vs facility` definitions may all be mixed together. Capacity, density, and efficiency must be audited before cross-record comparison.

### 2.5 Label Constraints (quick view before execution)
1. Positive-class definition: `g3_strict_v1` (high-confidence evidence positives).
2. Paradigm: PU learning / ranking (strict positives vs all other unlabeled samples).
3. Hard input bans: `llm_*`, `accel_*`, `stage`, `type`, `level`, and `year` must not enter the main model.

## 3. Record Granularity and Identifiers
### 3.1 What One Row Represents
R1 contains mixed-granularity records rather than a single "data hall building" definition. A brief explanation of `level` and `parent` is shown below.

| Field | Recommended Use | Main Limitation |
|---|---|---|
| `id` | Record primary key (unique within the current file) | Only guaranteed unique in the current snapshot; cross-version use requires versioned mapping. |
| `name` | Manual verification and evidence trace-back | Naming style is inconsistent and may include marketing names or phase names. |
| `company` | Grouped analysis and company-holdout splitting | Should not be used as a main-model feature; it can easily encode company disclosure and scale shortcuts. |
| `location` | Geographic stratification and manual inspection | Place-name granularity is inconsistent (city/state/country mixed together). |
| `type` | Metadata and stratified description | Disclosure definitions are mixed; it is a banned field for the main model in the methodology. |
| `year` | Temporal background description | High missingness (68.1%), and it may contain planning years (up to 2030), so it is not equivalent to "year in operation"; banned from the main model. |
| `stage` | Metadata / project-stage reference | Discrete values include `-1/0/1/2/3` and missing values, with inconsistent cross-source definitions; banned from the main model. |
| `level` | Record granularity marker (`site/campus/facility`) | Mixed granularity prevents direct horizontal capacity comparison; banned from the main model. |
| `parent` | Auxiliary parent-link field | High missingness and mixed ID sources; usable only as a weak linkage clue, not a complete hierarchy tree. |

The hard methodology constraints (R1_2 scheme) are consistent with the above: the main model excludes `stage`, `type`, `level`, and `year`, and also excludes `llm_*` and `accel_*` as inputs.

## 4. Field Dictionary (grouped by: Power / Racks / Density / Cooling / Space / Metadata)
### 4.1 Power
Field: `power_mw`  
Physical meaning: usually the available or planned IT power scale of a site/campus (unit: MW).  
Observed coverage (R1): 90.8% non-missing. The typical scale is in the "tens to hundreds of MW" range; the median is about 50 MW (this value may change with cleaning iterations).  
Common reasons for missingness: capacity not disclosed, only total utility/facility power disclosed rather than IT-load power, or campus-level and single-building definitions mixed together.  
Risk note: extreme values / outliers exist and may come from mixed granularity, mixed definitions, or mixed units; they must be audited before comparison.  
Relation to the target orientation (cautious): high power may be one necessary condition, but this orientation cannot be determined from a single field alone.

### 4.2 Racks
Fields: `rack_count`, `rack_kw_typical`, `rack_kw_peak`  
Physical meaning:
1. `rack_count`: rack scale (count).
2. `rack_kw_typical`: typical rack power density (kW/rack, normal operation).
3. `rack_kw_peak`: peak / upper-limit rack power density (kW/rack, instantaneous or design maximum).

Observed coverage (R1):
1. `rack_count`: 7.0% non-missing, the sparsest field; among disclosed samples, the scale is often around `10^2-10^3` (median about 1000, subject to cleaning changes).
2. `rack_kw_typical`: 54.3% non-missing, often at the level of tens of kW/rack (median about 50).
3. `rack_kw_peak`: 21.0% non-missing, usually higher than `typical` (median about 200).
4. `typical` vs `peak`: the former is closer to day-to-day operation, while the latter is closer to the design upper limit; they should not be mixed.
5. The medians above are only for scale illustration and may change with cleaning iterations.

Common reasons for missingness: vendors disclose only "supported density" without a typical value, disclose only a single marketing peak value, or mix site-level and campus-level entries.  
Risk note: this field group contains mixed definitions and outliers; granularity and units must be standardized before cross-record comparison.  
Relation to the target orientation (cautious): rack power density is related to compatibility, but it must be interpreted jointly with power, thermal management, and space.

### 4.3 Density
Field: `rack_density_area_w_per_sf_dc`  
Physical meaning: area-normalized power density (the field name indicates W/ft²).  
Observed coverage (R1): 15.5% non-missing; the typical magnitude is around `10^2` W/ft² (median about 165, subject to cleaning changes).  
Common reasons for missingness: inconsistent area definitions (GFA vs whitespace), unstandardized unit conversion, and many sources not directly disclosing this metric.  
Risk note: this field has high missingness and contains extreme values / outliers, suggesting possible definition or unit mixing; definition auditing is required before cross-record comparison.  
Relation to the target orientation (cautious): higher area power density may be a clue, but it is not sufficient evidence on its own.

### 4.4 Cooling / Thermal
Fields: `cooling`, `liquid_cool`, `pue`  
Physical meaning:
1. `cooling`: cooling-scheme category (such as `air`, `water_based_air`, `hybrid_air_liquid`, `liquid_direct_or_loop`, `liquid_immersion`).
2. `liquid_cool`: liquid-cooling capability marker (`Y/N/empty`).
3. `pue`: efficiency metric (Power Usage Effectiveness, usually >1).

Observed coverage (R1):
1. `cooling`: 79.2% non-missing, but still includes `unknown/empty`.
2. `liquid_cool`: 65.0% non-missing, with mixed `Y/N/empty`.
3. `pue`: 46.6% non-missing, with typical values around 1.1-1.3 (median about 1.17, subject to cleaning changes).

Common reasons for missingness: vendors disclose only "supports liquid cooling" without technical details, PUE is reported only as a design value or best-case value, and different climate zones are not directly comparable.  
Risk note: design values, marketing values, and operational values may be mixed, and regional conditions further limit comparability.  
Relation to the target orientation (cautious): liquid cooling and lower PUE often align with high-density deployment compatibility, but they do not by themselves verify the target orientation.

### 4.5 Space
Fields: `building_sqm`, `whitespace_sqm`  
Physical meaning:
1. `building_sqm`: total building area (often total gross floor area / GFA, unit: m²).
2. `whitespace_sqm`: IT-usable white-space area (unit: m²).

Observed coverage (R1):
1. `building_sqm`: 59.9% non-missing, commonly around `10^4` m² (median about 22,675, subject to cleaning changes).
2. `whitespace_sqm`: 40.3% non-missing, commonly around `10^3-10^4` m² (median about 7,200).

Common reasons for missingness: only land/campus scale is disclosed, phase 1 and total build-out are not distinguished, or white-space definitions differ (with or without MEP/support areas).  
Risk note: when campus-level and single-building area values are mixed, space fields can materially distort comparability.  
Relation to the target orientation (cautious): space fields do not represent the target orientation by themselves; they should be analyzed jointly with power, thermal management, and density.

### 4.6 Metadata
Fields: `id`, `name`, `company`, `location`, `type`, `year`, `stage`, `parent`, `level`  
Uses: retrieval, stratification, auditing, manual review, and sample splitting.  
Restriction: metadata should not be treated as "physical mechanism features"; in particular, `type/year/stage/level` are hard-banned main-model fields in the methodology.

## 5. Labels, Evidence Fields, and Usage Constraints
### 5.1 Label Fields (LLM judgment output)
Fields: `llm_ai_dc_label`, `llm_ai_dc_confidence`  
Meaning:
1. `llm_ai_dc_label`: the AI-related category label assigned by the LLM based on evidence text (R1 observes `ai_specific`, `ai_optimized`, `ai_capable_marketing`, `non_ai`, `ai_label`).
2. `llm_ai_dc_confidence`: the confidence score for the corresponding label (0.30 to 0.92).

Risk: this label is an "evidence + model judgment" product rather than ground truth, and it is affected by prompt wording, source-text quality, and marketing narratives.

### 5.2 Evidence-enrichment Fields (anchor candidates)
Fields: `accel_vendor`, `accel_model`, `accel_count`  
Role: used for manual verification, strict-positive anchor construction, and evidence-chain reinforcement.  
Limitation: disclosure bias is obvious (`accel_count` is 91.8% missing and `accel_model` is 54.8% missing).

### 5.3 Usage Constraints (mandatory)
1. The main model must be `infra-only`; `llm_*` and `accel_*` must not be used as input features.
2. `stage/type/level/year` also must not enter the main model.
3. `llm_*` and `accel_*` may be used to construct "high-confidence positive anchors" or support manual verification, but must not be used for leakage-driven score inflation.
4. In practice, the framing is: strict positive anchors (high-confidence evidence positives) vs all other unlabeled samples, using a PU/ranking perspective; `g3_strict_v1` is the role description for this definition.

## 6. Missingness & Noise: Known Issues and Risks
### 6.1 Most Severe Missing Fields (R1 observation)
1. `rack_count`: 93.0% missing.
2. `accel_count`: 91.8% missing.
3. `parent`: 90.1% missing.
4. `rack_density_area_w_per_sf_dc`: 84.5% missing.
5. `rack_kw_peak`: 79.0% missing.
6. `year`: 68.1% missing.
7. `whitespace_sqm`: 59.7% missing.
8. `pue`: 53.1% missing.

### 6.2 Why Missingness Is "Not Random"
Missingness in R1 is more likely to come from systematic disclosure differences rather than random loss:
1. Companies have different disclosure strategies (listed companies / large platforms more often disclose efficiency and advanced cooling).
2. Source types differ (press releases, marketing pages, and technical white papers vary greatly in information granularity).
3. Lifecycle stages differ (planned projects often disclose total power but not white space or actual rack fit-out).
4. Granularity differs (`campus` and `standalone_site` are naturally misaligned in capacity definitions).

### 6.3 Shortcut Risk and the Need for Auditing
The missingness pattern itself may become a model shortcut, meaning the model learns "who tends to disclose" rather than "who is more likely to exhibit the target orientation."  
Therefore, if missing-indicator variables are allowed, they must be paired with missingness ablation and audit linkage (including controlled missingness, negative control, and stratified stability checks), so that disclosure patterns are not mistaken for physical regularities.

## 7. Why Multi-field Combinations and Interactions Are Considered (and Why Guardrails)
### 7.1 Why Combinations Are Needed Instead of Single Fields
A single field is usually insufficient for an interpretable and generalizable judgment. Signals closer to physical mechanism usually arise from joint conditions, for example:  
`high power + high rack power density + liquid-cooling capability + reasonable space constraints`.

### 7.2 The Role of Interactions and Multi-field Combinations (controlled)
1. Interactions are used as explanatory clues for mechanism hypotheses and rulebook summaries, not as default score-boosting tools.
2. They may be elevated to generalizable rule conclusions only when support thresholds and stability / negative-control / controlled-missingness audits all pass together.
3. Missingness patterns must not be treated as interaction laws in themselves.

## 8. Practical Interpretation Guide (What You Can Conclude / Cannot Conclude)
### 8.1 Recommended Interpretation Framework (principles)
1. Prioritize "combined mechanisms" over the high/low value of any single field.
2. Combination patterns must pass stability and negative-control / controlled-missingness audits.
3. Do not treat `__MISSING__` as a transferable rule body.
4. Do not use `llm_*` / `accel_*` as main-model inputs, to avoid label and evidence leakage.

### 8.2 Two-layer Output Framing
1. `Score`: used for ranking and prioritization, answering "who should be reviewed first."
2. `Rulebook`: used for auditable explanation, answering "why this record is ranked highly."

### 8.3 Conclusions You Can and Cannot Draw
Can:
1. Identify the priority of objects that are more likely to exhibit the AI-oriented / accelerator-dense / high power-density compatible orientation.
2. Produce traceable explanatory clues and support manual review.

Cannot:
1. Interpret this database directly as a database of AI-load share or actual GPU utilization.
2. Treat the LLM labels as noiseless ground truth.
3. Claim interactions or missingness patterns to be generalizable laws without a supporting audit chain.
