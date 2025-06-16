# 🎬 IMDB Sentiment Analysis using Logistic Regression

This project performs **Sentiment Analysis** on IMDB movie reviews using machine learning techniques. The goal is to classify whether a given review is **positive** or **negative**.

## 📂 Dataset
The dataset used is from Kaggle:  
[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Each row in the dataset contains:
- A movie review (`review`)
- A sentiment label (`positive` or `negative`)

---

## 🧠 What This Project Does

- Cleans and preprocesses movie review text
- Removes stopwords, punctuation, and HTML
- Lemmatizes words (reduces to base form)
- Converts text into numerical vectors using **TF-IDF**
- Trains a **Logistic Regression** model to predict sentiment
- Evaluates model performance using accuracy, confusion matrix, and classification report

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK (Natural Language Toolkit)

---

## 🚀 How to Run the Project

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

### 2. Install dependencies
Make sure you have Python 3.x installed.

```bash
pip install pandas numpy scikit-learn nltk
```


### 3. 📥 Download the Dataset

Download the IMDB dataset from Kaggle and place the `IMDB Dataset.csv` file in the project folder.

📎 [Kaggle Dataset Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

### 4. ▶️ Run the Notebook or Script

#### If you’re using **Jupyter Notebook**:
```bash
jupyter notebook imdb_sentiment_analysis.ipynb
```

