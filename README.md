
# Project : Offensive Language Classification

## Project Overview

This project addresses the problem of **toxic and offensive language detection** in online user feedback. Given a comment, the task is to classify whether the comment is **toxic (offensive)** or not, based on various fine-grained labels such as `toxic`, `abusive`, `vulgar`, `menace`, `offense`, and `bigotry`.

While the training dataset contains multiple labels per comment (multi-label), the **evaluation is based on a binary classification**: whether the comment is overall offensive (1) or not (0). The project must also handle **multilingual text**, as the test and validation sets contain comments in multiple languages.

---


## Dataset Description – Offensive text Classification

This project uses a multilingual dataset to detect offensive content in online feedback. The data is split into three primary sets: **Training**, **Validation**, and **Test**.

---

###  1. `train.csv` – Labeled Training Data
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
  - `id`: Unique identifier
  - `feedback_text`: Feedback in multiple languages
  - `lang`: Language code
  - `toxic`: Binary label (1 = offensive, 0 = clean)

> Used to validate the model on **multilingual inputs**. The target label is `toxic`, which aligns with the binary `offensive` label in training.

---

### 3. `test.csv` – Multilingual Test Set (Unlabeled)
  - `id`: Unique identifier
  - `content`: Feedback to classify
  - `lang`: Language code

> Used for **prediction only**. Ground truth labels are provided separately.

---

### 4. `test_labels.csv` – Ground Truth for Test Set
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
---
---

# Model Implementation Details

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



Performance comparison and plots are included in both notebooks.
---


# Steps to Run

Prefering to run the notebook/pipelines in **Google Colab** or Jupiter notebook:

---

### Choose a Notebook

Navigate to the `task/` directory and open one of the following notebooks:

- **`model1_implementation.ipynb`**: Logistic Regression and LSTM models.
- **`model2_implementation.ipynb`**: Transformer-based model (BERT).

---

### Upload Dataset

Upload the required files (`train.csv`, `validation.csv`, `test.csv`, `test_labels.csv`) to Google Drive (if using Colab) or your local directory.  
 **Update file paths in the notebook** (e.g., `/content/drive/MyDrive/...`).

---

### Run the Notebook

- In Colab: **Runtime → Run all**  
- Locally: Run cells sequentially.

This will process the data, train models (or load saved models), and display metrics.

---

### Use Saved Models (Optional)

To save time, load pre-trained models. In this case, skip the model fiting codes.

---

The notebook will generate:

- Classification Reports, Accuracy, AUC-ROC, Confusion Matrix, ROC Curves 

---
--- 

# Model Evaluation Results

### Some observations : 
- **Class Imbalance:** The dataset is highly imbalanced — toxic labels are underrepresented. This affected the recall of simpler models.
- **Multilingual Challenge:** Validation and test sets include comments in multiple languages, but the training set is mainly English. BERT’s multilingual design helped significantly here.
- **Feature Engineering:** No custom preprocessing was needed for BERT. Traditional models required lemmatization, stopword removal, and TF-IDF.
- **Training Time:** Logistic Regression was the fastest to train. LSTM required tuning. BERT was slowest but offered the best performance.
- **Reusable Models:** All trained models were saved and can be reloaded for future inference without retraining.
- **Visualization:** Confusion Matrices, ROC Curves, and Word Distributions were generated for better model interpretability.
- **Generalization:** Fine-tuned BERT showed stable results even on unseen multilingual feedback, making it suitable for real-world deployment.

This project evaluates three types of models for detecting offensive or toxic content:

---

### 1. Baseline Model: Logistic Regression (TF-IDF)

- **Accuracy:** 0.770
- **AUC-ROC:** 0.540
- **Strengths:** Simple and fast; works well with clean and clearly separable text.
- **Limitations:** Struggles with complex, multilingual, or context-dependent phrases. Poor minority class recall.

---

### 2. Advanced Model: LSTM (Tuned)

- **Validation Accuracy:** 0.742
- **Validation AUC-ROC:** 0.509

- **Test Accuracy:** 0.718
- **Test AUC-ROC:** 0.457

- **Strengths:** Better sequence understanding than Logistic Regression. Able to model word order.
- **Limitations:** Still struggles to capture nuanced toxicity in multilingual content. Takes longer to train.

---

### 3. Transformer-Based Model: Fine-tuned BERT

- **Validation Accuracy:** 0.848
- **Validation AUC-ROC:** 0.598
- **Test Accuracy:** 0.793
- **Test AUC-ROC:** 0.610

- **Strengths:** Best performance across all metrics. Handles multilingual and context-aware inputs well.
- **Limitations:** Requires more computational resources. Training is slower than traditional models.

**Best Model:** BERT outperformed all other models, especially in detecting offensive content from multilingual inputs and edge cases. It is the most robust and generalizable solution for this task.

---


---

Author : Mizbah Uddin Junaed
