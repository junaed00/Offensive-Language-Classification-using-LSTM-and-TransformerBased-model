## Dataset Description – Offensive text Classification

This project uses a multilingual dataset to detect offensive content in online feedback. The data is split into three primary sets: **Training**, **Validation**, and **Test**.

---

###  1. `train.csv` – Labeled Training Data
- **Language:** English only  
- **Size:** ~23,000 rows  
- **Columns:**
  - `id`: Unique identifier
  - `feedback_text`: The comment to classify
  - `toxic`: 1 if toxic
  - `abusive`: 1 if abusive
  - `vulgar`: 1 if vulgar
  - `menace`: 1 if threatening
  - `offense`: 1 if offensive
  - `bigotry`: 1 if identity-based hate

- **Derived Label:**
  - `offensive`: A binary label calculated using the max value across the above 6 columns.  
    `offensive = max(toxic, abusive, vulgar, menace, offense, bigotry)`

>  Used for model training. Although the dataset contains multiple fine-grained labels, the model is trained to detect **binary offensive** content.

---

###  2. `validation.csv` – Multilingual Validation Set
- **Language:** Multilingual (e.g., `en`, `fr`, `tr`, etc.)
- **Size:** 840 rows  
- **Columns:**
  - `id`: Unique identifier
  - `feedback_text`: Feedback in multiple languages
  - `lang`: Language code
  - `toxic`: Binary label (1 = offensive, 0 = clean)

> Used to validate the model on **multilingual inputs**. The target label is `toxic`, which aligns with the binary `offensive` label in training.

---

### 3. `test.csv` – Multilingual Test Set (Unlabeled)
- **Language:** Multilingual
- **Size:** ~6,000 rows  
- **Columns:**
  - `id`: Unique identifier
  - `content`: Feedback to classify
  - `lang`: Language code

> Used for **prediction only**. Ground truth labels are provided separately.

---

### 4. `test_labels.csv` – Ground Truth for Test Set
- **Size:** ~6,000 rows (subset of test set)  
- **Columns:**
  - `id`: Matches `test.csv`
  - `toxic`: Binary ground truth (1 = offensive, 0 = not offensive)

> Used to evaluate test set performance. This allows comparison of predicted and actual values.

---

### Summary Table

| File              | Language     | Purpose         | Label Type                       | Rows   |
|-------------------|--------------|------------------|----------------------------------|--------|
| `train.csv`        | English       | Training         | Multi-label + binary `offensive` | ~23,000 |
| `validation.csv`   | Multilingual  | Validation       | Binary `toxic`                   | 840    |
| `test.csv`         | Multilingual  | Test Prediction  | No labels                        | ~6,000 |
| `test_labels.csv`  | Multilingual  | Test Evaluation  | Binary `toxic`                   | ~6,000    |

---

**Note:**  
The main prediction target is a **binary label** (offensive = 1 / not offensive = 0).  
Fine-grained labels enrich training but are not directly evaluated.

