# Project_truth_miner


👀 **What is TruthMiner?**

In today’s world, misinformation travels faster than facts.  
TruthMiner is an AI-powered system that detects fake news headlines using both classical Machine Learning and modern Transformer models (**DistilBERT**).  
Think of it as your digital truth detector — analyzing headlines and predicting whether they’re real or fake.

---

🚀 **Project Overview**

TruthMiner is an NLP project designed to automatically classify news headlines as either real or fake.  
It combines classical ML models and a fine-tuned Transformer to benchmark performance and highlight trade-offs between traditional approaches and modern deep learning.

---

🎯 **Goals**

- Preprocess raw text into clean, usable features
- Apply feature engineering with TF-IDF and embeddings
- Train and compare multiple ML classifiers:
  - Logistic Regression
  - Naïve Bayes
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- Fine-tune **DistilBERT** for sequence classification
- Evaluate models using Accuracy, Precision, Recall, F1-score
- Generate predictions for the test dataset in the original file format

---

📂 **Dataset**

- **Training data:** `dataset/training_data.csv`  
  Format: `label<TAB>headline`  
  `label = 0 → Fake News`  
  `label = 1 → Real News`  

- **Testing data:** `dataset/testing_data.csv`  
  Same format, but labels are placeholders (2). Your model predicts and replaces them with 0 or 1.

---

⚙️ **Workflow**

**1️⃣ Preprocessing**

- Convert to lowercase  
- Remove punctuation  
- Tokenize text  
- Remove stopwords  
- Apply stemming / lemmatization  

**2️⃣ Feature Engineering**

- TF-IDF vectorization (unigrams + bigrams)  
- Embeddings for Transformer input  

**3️⃣ Modeling**

- Train classical ML models with TF-IDF features  
- Fine-tune DistilBERT using Hugging Face Trainer API  
- Apply stratified splits (80/20) for validation  

**4️⃣ Evaluation**

- Metrics reported: Accuracy, Precision, Recall, F1-score  
- Calculate **Overfit Gap** (train vs validation accuracy)

**5️⃣ Prediction**

- Replace placeholder labels in the test set with predicted values  
- Save results as `testing_predictions.csv` (tab-separated, no header)

---

📊 **Results & Overfitting Analysis (Validation Set)**

| Model                  | 🟢 Validation | 🟡 Train | ⚠️ Overfit Gap | ✨ Precision | 🔹 Recall | 🏆 F1 |
|------------------------|---------------|----------|----------------|-------------|-----------|-------|
| Linear SVM             | 0.9303        | 0.9753   | 0.04496        | 0.9178      | 0.9407    | 0.9291|
| Logistic Regression    | 0.9271        | 0.9510   | 0.02388        | 0.9094      | 0.9439    | 0.9263|
| Naive Bayes            | 0.9228        | 0.9396   | 0.01684        | 0.9278      | 0.9119    | 0.9198|
| Random Forest          | 0.9138        | 0.99996  | 0.08614        | 0.9072      | 0.9162    | 0.9117|
| XGBoost                | 0.8729        | 0.8894   | 0.01643        | 0.8154      | 0.9544    | 0.8794|
| DistilBERT (fine-tuned)| 0.9600        | 0.9800   | 0.0200         | 0.9600      | 0.9600    | 0.9600|

📌 **Notes:** DistilBERT slightly outperformed classical ML models, but SVM and Logistic Regression were strong baselines.

---

**📊 Key Insights**
- DistilBERT slightly outperformed classical ML models in both accuracy and F1-score.  
- SVM and Logistic Regression were strong baseline models with minimal overfitting.  
- Random Forest showed the highest overfitting gap, despite high training accuracy.  
- Feature engineering (TF-IDF + embeddings) was crucial to improve performance across all models.  

---

💡 **Conclusion**
- Combining classical ML with Transformers allows benchmarking trade-offs between interpretability and performance.  
- DistilBERT achieved the highest validation accuracy and F1, demonstrating the advantage of modern NLP architectures.  
- Overfitting analysis highlighted which models generalize better to unseen data.  
- Future improvements: expand dataset, experiment with larger Transformer models, and implement ensemble strategies to boost robustness.

  ---
  
🙋🏽‍♀️ **About Me**

- **Rocío Zahory Vásquez Romero**  
- Senior Auditor | Data Science & Machine Learning Enthusiast  
- Email: rocio.vasquez@usach.cl  
- LinkedIn: [https://www.linkedin.com/in/rocio-zahory-vasquez-romero-3621ab1a7/](https://www.linkedin.com/in/rocio-zahory-vasquez-romero-3621ab1a7/)

 ---
 
⚡ **How to Run**

1. Clone the repo:  
```bash
git clone https://github.com/yourusername/truthminer.git
cd truthminer

2. Create and activate a virtual environment (recommended):
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

3.Install dependencies:
pip install -r requirements.txt

4. Run the main notebook: project_truth_miner.ipynb
