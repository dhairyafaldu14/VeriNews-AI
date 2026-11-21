# VeriNews AI — Intelligent News Classification System

VeriNews AI is a machine-learning-based news text classification project. It predicts whether a submitted news headline and article body are **Real News** or **Fake News** using TF-IDF features and a trained Linear Support Vector Machine (Linear SVM).

> **Important:** VeriNews AI is a text classifier, not an independent fact-checking system. It learns patterns from the WELFake dataset. It does not verify claims against reliable sources or establish whether an event actually happened.

## Project objectives

- Clean and prepare a labelled news dataset.
- Convert news text into numerical features using TF-IDF.
- Train and compare multiple machine-learning classifiers.
- Evaluate the selected model on held-out data.
- Build a Streamlit interface to demonstrate predictions and model explanations.

## Dataset

This project uses the **WELFake dataset** (`WELFake_Dataset.csv`), which contains news articles labelled as real or fake.

The notebook's workflow includes:
- Inspecting and handling missing or empty values.
- Removing duplicate combined article content.
- Combining headline and article body for model input.
- Splitting data into training and testing sets while grouping identical article bodies to reduce train/test overlap.
- Fitting TF-IDF on training text only and transforming the test text separately.

The final deduplicated dataset used in the recorded workflow contained **63,108 rows**.

## Methodology

### Text preprocessing and feature extraction

The notebook combines the headline and article body into one text input. It uses `TfidfVectorizer` with:

- `max_features=50000`
- `ngram_range=(1, 2)` — unigrams and bigrams
- `stop_words='english'`

The vectorizer is fitted on the training data only to avoid fitting vocabulary on the test set.

### Machine-learning models

The notebook compares:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (Linear SVM)

The Linear SVM was selected based on the recorded evaluation results.

## Model evaluation

Recorded results for the selected Linear SVM on the held-out WELFake test split:

| Metric | Result |
|---|---:|
| Accuracy | 96.07% |
| Macro F1-score | 0.96 |
| ROC-AUC | 0.9926 |
| Average Precision | 0.9915 |

These are results from the project's held-out dataset split. They are not a guarantee of performance on current news, other datasets, or real-world claims. The external evaluation conducted during development was small and exploratory, so it should not be presented as a representative benchmark.

## Streamlit application

The Streamlit app (`app.py`) accepts:
- A news headline
- The news article body

It then displays:
- Predicted category: Real News or Fake News
- Linear SVM decision score
- Up to five influential TF-IDF features contributing toward the predicted class
- A disclaimer about the model's limitations

### Understanding the SVM decision score

The decision score is **not a probability**. A positive score indicates the Fake News class, while a negative score indicates the Real News class. Its magnitude represents distance from the decision boundary, not a probability or percentage confidence.

The influential words are model feature contributions. They are not evidence that an article is true or false.

## Project structure

The complete project folder should contain:

```text
VeriNews-AI/
├── VeriNews_AI.ipynb
├── WELFake_Dataset.csv
├── app.py
├── verinews_tfidf.pkl
├── verinews_svm_model.pkl
├── requirements.txt
└── README.md
```

The CSV dataset is needed to reproduce the notebook's training and preprocessing workflow. The deployed Streamlit app uses the two saved `.pkl` files and does not need to load the CSV.

## Run the Streamlit app locally

1. Install Python.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the app:

   ```bash
   streamlit run app.py
   ```

6. Open the local address printed by Streamlit in your browser.

Keep `app.py`, `verinews_tfidf.pkl`, and `verinews_svm_model.pkl` in the same directory. The app loads the model files using these filenames.

## Deployment

The app can be deployed through Streamlit Community Cloud by connecting the GitHub repository and selecting `app.py` as the main file. The repository must include `app.py`, both model files, and `requirements.txt`.

The dataset and notebook can also be included in the repository for project transparency and reproducibility, subject to GitHub's file-size limits.

## Technologies and libraries

- Python
- Pandas
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit

The notebook was developed in Google Colab. The requirements file includes the libraries used by the notebook and app. Google Colab-specific utilities are provided by the Colab environment and are not required to run the Streamlit app locally.

## Limitations

- The model learns statistical text patterns from its training dataset; it does not independently verify facts.
- Predictions can be affected by topic, writing style, dataset bias, and changes in news language over time.
- A high held-out test score does not guarantee equivalent real-world performance.
- The SVM decision score is not a probability.
- Feature contributions explain which learned model features influenced a prediction; they do not establish factual evidence.

For important news, verify claims using multiple reliable and independent sources.
