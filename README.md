
# Project : Offensive Language Classification

## Project Overview

This project addresses the problem of **toxic and offensive language detection** in online user feedback. Given a comment, the task is to classify whether the comment is **toxic (offensive)** or not, based on various fine-grained labels such as `toxic`, `abusive`, `vulgar`, `menace`, `offense`, and `bigotry`.

While the training dataset contains multiple labels per comment (multi-label), the **evaluation is based on a binary classification**: whether the comment is overall offensive (1) or not (0). The project must also handle **multilingual text**, as the test and validation sets contain comments in multiple languages.

---


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
  - `offensive`: A binary label for the overal offensiveness (0 or 1)

>  Used for model training. Although the dataset contains multiple fine-grained labels, the model is trained to detect **binary overal offensive** content.

---

###  2. `validation.csv` – Multilingual Validation Set
- **Language:** Multilingual (e.g., `en`, `fr`, `tr`, etc.)
- **Columns:**
  - `id`: Unique identifier
  - `feedback_text`: Feedback in multiple languages
  - `lang`: Language code
  - `toxic`: Binary label (1 = offensive, 0 = clean)

> Used to validate the model on **multilingual inputs**. The target label is `toxic`, which aligns with the binary `offensive` label in training.

---

### 3. `test.csv` – Multilingual Test Set (Unlabeled)
- **Language:** Multilingual
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


## Model Implementation Details

### 1. Data Processing

- **Datasets Used**:
  - `train.csv` → English-only labeled data with 6 fine-grained labels.
  - `validation.csv` → Multilingual data with a binary `toxic` label.
  - `test.csv` → Unlabeled multilingual data for prediction.
  - `test_labels.csv` → True binary labels for the test set.

- **Preprocessing Steps**:
  - For traditional ML and LSTM: Lowercasing, removal of URLs/special characters, lemmatization, stopword removal.
  - For BERT: No manual cleaning (uses raw text), handled via tokenizer.
  - Feature extraction:
    - TF-IDF (for Logistic Regression)
    - Tokenization + Padding (for LSTM)
    - Transformer embeddings (BERT-based model)

---

### 2. Baseline Model (Logistic Regression)

- **Text representation**: TF-IDF (with unigrams and bigrams)
- **Model**: `LogisticRegression` from `sklearn` with `class_weight='balanced'` for imbalance handling
- **Evaluation**: Accuracy, F1-score, AUC-ROC, Confusion Matrix on both validation and test sets.

---

### 3. Advanced Model (LSTM)

- **Tokenizer**: `keras.preprocessing.text.Tokenizer`
- **Architecture**:
  - Embedding Layer
  - LSTM (with `return_sequences=True`)
  - GlobalMaxPooling
  - Dense + Dropout Layers
- **Tuning**:
  - Hyperparameter tuning done using **Keras Tuner** (Random Search)
  - Tuned hyperparameters include: `embedding_dim`, `lstm_units`, `dropout`, `dense_units`, and `optimizer`

- **Best model was saved** to `.h5` format and used later for predictions to avoid rerunning long training steps.

---

### 4. Transformer-Based Model (BERT)

- **Model**: `TFBertForSequenceClassification` from Hugging Face Transformers
- **Tokenizer**: `BertTokenizer` with `bert-base-multilingual-cased`
- **Strategy**:
  - Fine-tuned on the training set using binary `offensive` labels
  - Evaluated on multilingual validation and test sets using the `toxic` label
  - Trained with `SparseCategoricalCrossentropy` and `Adam` optimizer with learning rate `2e-5`
  - Used class weights to address class imbalance

- **Saved and reused** the fine-tuned model and tokenizer using `.save_pretrained()` and `.from_pretrained()` for efficiency.

---

### Evaluation

Each model was evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- AUC-ROC Curve

Performance comparison and plots are included in both notebooks.

---
By,
Mizbah Uddin Junaed
