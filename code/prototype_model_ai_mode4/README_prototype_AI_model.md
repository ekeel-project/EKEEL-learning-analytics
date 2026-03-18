# Tutor AI — Learning Analytics Pipeline
# This is an extension of the EKEEL project results providing a preliminary model toward an adaptive tutoring framework, by estimating learners’ expected final performance in real time, supporting adaptive tutoring interventions based on interaction dynamics and concept-level behavior.

## What this pipeline does
- log parsing and session reconstruction from VR learning logs
- normalization of action and concept identifiers
- extraction of behavioral and performance features from interaction sequences
- creation of training samples at each `PlayAnnotation` event
- supervised learning model for predicting the **final session score**
- deterministic curriculum-based concept progression
- real-time prediction of expected final score during a session
- interactive dashboard for log navigation and prediction inspection
- export of a reusable model bundle (`.joblib`) for inference

The system simulates an **AI tutor assistant** capable of estimating the expected final performance of a learner based on their current interaction state.

---

# Repository structure

```
project/
│
├─ train_tutor_model.py        # training pipeline
├─ tutor_dashboard.py          # Streamlit dashboard for inference
│
├─ logs/                       # input log files
│
├─ models/
│   └─ tutor_model.joblib      # trained model bundle
│
└─ outputs/                    # optional analysis outputs
```

---

# Training pipeline

Script: `train_tutor_model.py`

## What it does

The training script builds a supervised dataset from interaction logs and trains a regression model.

For each `PlayAnnotation` event it:

1. reconstructs the **session state up to the previous event**
2. extracts behavioral and interaction features
3. records the annotation ID (`ann_id`) shown to the learner
4. assigns the **final session score** (from `ScoreSummary`) as the target variable

The model therefore learns:

> Given the current learner state and a candidate annotation, what final score is expected?

The trained model can then estimate how different instructional decisions might influence the final learning outcome.

---

## Extracted features

Examples of computed indicators include:

### Interaction counts
- number of `Touch`
- number of `Grab`
- number of `Release`
- number of `Pause`
- number of `Teleport`
- number of `PlayAnnotation`

### Timing indicators
- elapsed time from session start
- median delay between `Touch` and `PlayAnnotation`
- pause frequency per minute
- teleport frequency per minute

### Behavioral indicators
- unique concepts touched
- entropy of actions after `Touch`
- longest pause streak
- idle gaps in interaction

### Learning performance
- number of `MatchAttempt`
- number of incorrect matches
- match accuracy
- concept solve time
- number of attempts per solved concept

### Context indicators
- last touched concept
- last solved concept
- number of interactions on map/table

---

## Model

The pipeline uses a **Scikit-learn pipeline** consisting of:

### Preprocessing
- numerical features  
  - median imputation  
  - standard scaling
- categorical features  
  - most frequent imputation  
  - one-hot encoding

### Regression model

```
HistGradientBoostingRegressor
```

The model predicts:

```
target_score_percent
```

which corresponds to the final score obtained at the end of the session.

---

## Training dataset generation

Training rows are generated for every `PlayAnnotation` event.

Example:

```
state_features + ann_id → predicted_final_score
```

Each row represents the **learning state at the moment when an annotation is played**.

---

## Run training

```bash
python train_tutor_model.py \
  --input_dir logs \
  --out_model models/tutor_model.joblib
```

### Parameters

| Parameter | Description |
|---|---|
| `--input_dir` | directory containing `.txt` or `.csv` log files |
| `--out_model` | output path for the trained model bundle |
| `--test_ratio` | ratio of sessions used for validation (default `0.2`) |

---

## Output

The training pipeline produces a model bundle:

```
tutor_model.joblib
```

This bundle contains:

```
pipeline
feature_cols
num_cols
cat_cols
candidate_anns
concept_to_ann
```

Where:

- `pipeline` = trained preprocessing + model
- `feature_cols` = ordered feature schema
- `candidate_anns` = annotations seen during training
- `concept_to_ann` = mapping between concepts and annotation IDs

---

# Tutor dashboard (inference)

Script: `tutor_dashboard.py`

The dashboard provides an interactive interface for analyzing logs and predicting learner outcomes.

---

## What it does

The dashboard allows the user to:

- load a trained tutor model
- load a VR learning session log
- navigate the log step by step
- reconstruct the **current learner state**
- suggest the **next concept in the curriculum**
- estimate the **final session score** if the suggested annotation is shown

---

## Prediction workflow

At any moment in the log:

1. the session is reconstructed up to the current row
2. interaction features are computed
3. the last solved concept is identified
4. the next concept is determined using a curriculum order
5. the concept is mapped to its annotation ID
6. the trained model predicts the **expected final score**

---

## Curriculum order

Concept progression follows a predefined learning order:

```
cloud_computing
servizio
software
risorse_hardware
fornitore_del_servizio
infrastruttura_cloud
livello_astratto
livello_fisico
modello_di_cloud
saas
paas
iaas
on_premises
caratteristica
```

The next concept is mapped to its annotation using:

```
concept_to_ann
```

stored in the trained model bundle.

---

## Run the dashboard

```bash
streamlit run tutor_dashboard.py
```

The interface will ask for:

- path to the trained model (`.joblib`)
- path to a log file or a log directory

---

## Dashboard features

The Streamlit interface provides:

### Navigation controls
- move forward/backward through log rows
- jump to the next `MatchAttempt`
- slider navigation across the session timeline

### Indicators
- current score observed in the session
- predicted final score

### AI suggestion
- last solved concept
- next concept in the curriculum
- recommended annotation ID

### Debug tools
- feature vector used by the model
- state indicators up to the current log row
- concept-to-annotation mapping stored in the model

---

# Expected log format

Each log entry must contain the following columns:

```
Timestamp, Action, Field1, Field2, Field3, Field4, Field5
```

Typical actions include:

```
Touch
Grab
Release
PlayAnnotation
MatchAttempt
ConceptSolve
Pause
Teleport
ScoreSummary
```

---

# Example workflow

### 1 — Train the tutor model

```bash
python train_tutor_model.py \
  --input_dir logs \
  --out_model models/tutor_model.joblib
```

### 2 — Launch the dashboard

```bash
streamlit run tutor_dashboard.py
```

### 3 — Load resources

Provide:

- the trained model
- a session log

### 4 — Inspect the session

The system will show:

- current learner indicators
- suggested next annotation
- predicted final score

---

# Notes

- The system estimates **final session performance**, not the next action.
- Curriculum progression is deterministic, while the predicted outcome is learned from data.
- The model can support future extensions such as adaptive tutoring or annotation ranking.