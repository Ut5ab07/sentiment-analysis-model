# Sentiment Analysis on Real-World Text Data using NLP and Machine Learning

## Project Overview

This project performs binary sentiment analysis on movie reviews using
Natural Language Processing (NLP) and machine learning.

The goal is to classify each movie review as either:

-   **Positive**
-   **Negative**

The project follows a complete NLP workflow, starting from raw review
text and ending with model evaluation and comparison.

------------------------------------------------------------------------

## Dataset

### IMDb Movie Reviews Dataset

The project uses the **IMDb Movie Reviews Dataset**, which contains:

-   **50,000 movie reviews**
-   **2 columns**
    -   `review` --- Movie review text
    -   `sentiment` --- Positive or negative sentiment

During data cleaning, **418 duplicate reviews were removed**, leaving
**49,582 unique reviews** for further processing.

### Why is the dataset not included in this repository?

The dataset is not stored in this GitHub repository because of its file
size and repository storage limitations.

Instead, the repository contains the code and documentation required to
reproduce the project. The dataset should be downloaded separately and
placed in the project directory before running the notebook.

Expected dataset filename:

``` text
IMDB Dataset.csv
```

The dataset is used only as the input data for the experiment and is not
modified or redistributed through this repository.

------------------------------------------------------------------------

## Project Workflow

``` text
IMDb Movie Reviews
        |
        v
Dataset Loading
        |
        v
Data Exploration
        |
        v
Remove Duplicate Reviews
        |
        v
Text Preprocessing
        |
        +--> Remove HTML Tags
        |
        +--> Convert to Lowercase
        |
        +--> Remove Punctuation and Numbers
        |
        +--> Tokenization
        |
        +--> Stopword Removal
        |
        +--> Lemmatization
        |
        v
Train-Test Split
        |
        v
TF-IDF Feature Extraction
        |
        v
Machine Learning Models
        |
        +--> Logistic Regression
        |
        +--> Multinomial Naive Bayes
        |
        +--> Support Vector Machine
        |
        v
Model Evaluation
        |
        +--> Accuracy
        +--> Precision
        +--> Recall
        +--> F1-Score
        +--> Confusion Matrix
        |
        v
Model Comparison
```

------------------------------------------------------------------------

## 1. Data Exploration

The dataset was first inspected to understand its structure, sentiment
distribution, missing values, and duplicate records.

The original dataset contained:

``` text
50,000 reviews
2 columns
```

A total of:

``` text
418 duplicate reviews
```

were identified and removed.

The final dataset contained:

``` text
49,582 unique reviews
```

------------------------------------------------------------------------

## 2. Text Preprocessing

The raw review text was cleaned before feature extraction.

The following preprocessing steps were performed:

### HTML Removal

HTML tags such as `<br />` were removed from the reviews.

### Lowercase Conversion

All text was converted to lowercase so that words such as `Good` and
`good` were treated as the same word.

### Punctuation and Number Removal

Punctuation, special characters, and numbers were removed.

### Tokenization

Each review was split into individual words or tokens.

### Stopword Removal

Common English stopwords were removed to reduce unnecessary terms.

### Lemmatization

Words were converted into their base form using NLTK's WordNet
lemmatizer.

------------------------------------------------------------------------

## 3. Train-Test Split

The cleaned dataset was divided into:

-   **80% training data**
-   **20% testing data**

The split used stratification so that the positive and negative
sentiment distribution was maintained across the training and testing
sets.

------------------------------------------------------------------------

## 4. TF-IDF Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert
the cleaned text into numerical features.

The vectorizer was configured with:

``` python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)
```

This means that the model used up to **10,000 features** and included
both:

-   Unigrams --- individual words
-   Bigrams --- pairs of consecutive words

TF-IDF gives higher importance to terms that are useful for
distinguishing documents while reducing the importance of terms that
occur frequently across the dataset.

------------------------------------------------------------------------

## 5. Machine Learning Models

Three classification algorithms were trained and compared.

### Logistic Regression

A linear classification algorithm used to predict whether a review
belongs to the positive or negative sentiment class.

### Multinomial Naive Bayes

A probabilistic classifier commonly used for text classification.

### Support Vector Machine

A linear SVM was used to classify the TF-IDF feature vectors into
positive and negative sentiment classes.

------------------------------------------------------------------------

## 6. Model Evaluation

The models were evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix

### Results

  Model                   Accuracy   Precision   Recall   F1-Score
  --------------------- ---------- ----------- -------- ----------
  Logistic Regression       88.94%      88.20%   89.99%     89.09%
  Naive Bayes               86.45%      85.72%   87.58%     86.64%
  SVM                       88.52%      88.29%   88.93%     88.61%

Logistic Regression achieved the highest accuracy and F1-score among the
three models in this experiment.

------------------------------------------------------------------------

## 7. Project Structure

``` text
sentiment-analysis/
│
├── Sentiment_Analysis.ipynb
├── README.md
│
└── dataset/
    └── IMDB Dataset.csv    # Not included in GitHub
```

If the notebook expects the dataset in the root directory instead, place
`IMDB Dataset.csv` in the same directory as the notebook.

------------------------------------------------------------------------

## 8. Technologies Used

-   Python
-   Jupyter Notebook
-   Pandas
-   NLTK
-   Scikit-learn
-   Matplotlib
-   Seaborn

------------------------------------------------------------------------

## 9. How to Run

### Step 1: Clone the repository

``` bash
git clone <your-github-repository-url>
cd sentiment-analysis
```

### Step 2: Install the required libraries

``` bash
pip install pandas nltk scikit-learn matplotlib seaborn
```

### Step 3: Download the IMDb dataset

Download the IMDb Movie Reviews Dataset separately and place:

``` text
IMDB Dataset.csv
```

in the location expected by the notebook.

### Step 4: Open the notebook

``` bash
jupyter notebook
```

Open:

``` text
Sentiment_Analysis.ipynb
```

### Step 5: Run the notebook

Run the cells in order:

1.  Dataset loading
2.  Dataset exploration
3.  Duplicate removal
4.  Text preprocessing
5.  Train-test split
6.  TF-IDF feature extraction
7.  Model training
8.  Prediction
9.  Evaluation
10. Confusion matrices and performance comparison

------------------------------------------------------------------------

## 10. Outputs

The project produces:

-   Dataset preview
-   Preprocessed review examples
-   TF-IDF feature representation
-   Predictions from all three models
-   Model performance comparison
-   Confusion matrices
-   Accuracy, precision, recall, and F1-score results

------------------------------------------------------------------------

## 11. Limitations

-   The project performs binary sentiment classification only.
-   It does not classify neutral or mixed sentiments.
-   The models depend on the quality and characteristics of the IMDb
    dataset.
-   TF-IDF does not capture the full semantic meaning or context of a
    sentence.
-   The dataset is not included in the repository because of its size.

------------------------------------------------------------------------

## 12. Future Improvements

Possible improvements include:

-   Using Word2Vec or other word embeddings
-   Using transformer-based models such as BERT
-   Adding neutral and mixed sentiment classes
-   Hyperparameter tuning
-   Trying additional classifiers
-   Building a web interface for real-time sentiment prediction
-   Deploying the trained model as an API

------------------------------------------------------------------------

## Author
Utsab Raj Acharya
