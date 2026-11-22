# VeriNews AI — Intelligent News Classification System

An AI-powered news classification application that uses Natural Language Processing (NLP) and Machine Learning to classify news articles as **Real News** or **Fake News**.

## 🚀 Live Demo

**Try VeriNews AI:** https://verinews-ai.streamlit.app/

## 📌 Project Overview

VeriNews AI is a machine learning project developed as part of the **AI & ML Internship at Elevate Labs**.

The system analyzes the textual content of news articles and predicts whether they belong to the Real News or Fake News category.

It uses TF-IDF for text feature extraction and a Linear Support Vector Machine (SVM) classifier for prediction.

The application also displays the SVM decision score and important influential words to help users understand the model's prediction.

## ✨ Features

- Real-time news classification.
- Real News and Fake News prediction.
- TF-IDF-based text feature extraction.
- Linear SVM classification model.
- SVM decision score visualization.
- Important influential words contributing to predictions.
- Interactive web interface built with Streamlit.

## 🛠️ Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine learning and NLP
- **TF-IDF Vectorizer** – Text feature extraction
- **Linear SVM** – News classification
- **Joblib** – Model serialization
- **Streamlit** – Web application development
- **Matplotlib** – Data visualization
- **Jupyter Notebook / Google Colab** – Model development

## 📂 Project Structure

```text
VeriNews-AI/
│
├── app.py
├── VeriNews_AI.ipynb
├── verinews_svm_model.pkl
├── verinews_tfidf.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset

The project uses the **WELFake Dataset**, which contains real and fake news articles.

Dataset information:

- Original dataset: 72,134 records
- Records after preprocessing and duplicate removal: 63,108
- Real News: 34,790
- Fake News: 28,318

The dataset was cleaned and preprocessed before model training.

The original dataset is not included in this GitHub repository because of its large file size.

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Removed unnecessary columns.
2. Handled missing values.
3. Combined article titles and text.
4. Removed duplicate articles.
5. Checked class distribution.
6. Split the dataset into training and testing sets.
7. Applied TF-IDF vectorization.

To prevent data leakage, articles with overlapping text bodies were kept within the same data split.

## 🤖 Model Development

Three machine learning algorithms were evaluated:

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 94.83% |
| Multinomial Naive Bayes | 87.79% |
| Linear Support Vector Machine | 96.07% |

**Selected Model: Linear Support Vector Machine (SVM)**

The Linear SVM model achieved the highest test accuracy and was selected for the final application.

### TF-IDF Configuration

- Maximum features: 50,000
- N-gram range: (1, 2)
- Stop words: English

## 📈 Model Performance

The final Linear SVM model achieved the following results on the held-out test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 96.07% |
| Macro F1-Score | 0.96 |
| ROC-AUC | 0.9926 |
| Average Precision | 0.9915 |

### Confusion Matrix

```text
[[6758, 227],
 [ 269, 5361]]
```

Class labels:
- 0 = Real News
- 1 = Fake News

The model was evaluated using a grouped train-test split to reduce the risk of duplicate article leakage.

## 🧠 Prediction Explanation

VeriNews AI provides additional information alongside its predictions.

### SVM Decision Score

- Positive score: Fake News
- Negative score: Real News
- Score close to zero: Prediction is closer to the decision boundary.

**Important:** The SVM decision score is not a probability or confidence percentage.

### Important Influential Words

The application identifies words and phrases that contribute toward the predicted classification.

These features explain the model's learned associations, but they do not independently establish whether a news article is true or false.

## 💻 Run Locally

Follow these steps to run VeriNews AI on your computer.

### 1. Clone the Repository

```bash
git clone https://github.com/dhairyafaldu14/VeriNews-AI.git
```

### 2. Navigate to the Project Directory

```bash
cd VeriNews-AI
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 📦 Model Files

The trained machine learning components are stored separately:

- `verinews_svm_model.pkl` – Trained Linear SVM classifier.
- `verinews_tfidf.pkl` – Fitted TF-IDF vectorizer.

These files are loaded by the Streamlit application to generate predictions.

## ⚠️ Limitations

- The model relies on patterns learned from its training dataset.
- It may produce incorrect predictions.
- It can be affected by misleading wording, unfamiliar topics, and changes in news language.
- Important influential words do not prove the truth or falsity of an article.
- It does not independently verify claims against trusted news sources.

**VeriNews AI is a text classification system, not a fact-checking system. Always verify important news through reliable sources.**

## 🎯 Project Objective

The objective of this project is to demonstrate how Natural Language Processing and Machine Learning can be used to classify news articles and provide interpretable prediction information through an interactive web application.

## 👨‍💻 Internship Information

**Program:** AI & ML Internship  
**Organization:** Elevate Labs  
**Project:** Task 6 – VeriNews AI: Intelligent News Classification System

---

**Developed as part of the Elevate Labs AI & ML Internship.**