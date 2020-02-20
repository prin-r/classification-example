from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import process_text
import pickle
import pandas as pd
import numpy as np

# filename
TFIDF = "tfidf.pkl"
MODEL = "model.pkl"

# load csv
all_df = pd.read_csv("data/data.csv")
test_df = pd.read_csv("data/test.csv")

# divide all_df into train_df(85%) and valid_df(15%)
train_df, valid_df = train_test_split(all_df, test_size=0.15, random_state=1412)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

# dependent variables
y_train = train_df["category"]
y_valid = valid_df["category"]
y_test = test_df["category"]

# text faetures
# Create Vectorizer and dump to pkl
tfidf = TfidfVectorizer(
    tokenizer=process_text, ngram_range=(1, 2), min_df=20, sublinear_tf=True
)
tfidf_fit = tfidf.fit(all_df["texts"])
pickle.dump(tfidf_fit, open(TFIDF, "wb"))
print("Dump ", TFIDF, " completed")

# transform word to vector(vectorize)
text_train = tfidf_fit.transform(train_df["texts"])
text_valid = tfidf_fit.transform(valid_df["texts"])
text_test = tfidf_fit.transform(test_df["texts"])
# print(text_train.shape, text_valid.shape)

# convert to array
X_train = text_train.toarray()
X_valid = text_valid.toarray()
X_test = text_test.toarray()

# fit logistic regression models
model = LogisticRegression(
    C=2.0, penalty="l2", solver="liblinear", dual=False, multi_class="ovr"
)
model.fit(X_train, y_train)

# dump model to model.pickle
pickle.dump(model, open(MODEL, "wb"))
print("Dump ", MODEL, " completed")

# test score
print("score_validate:", model.score(X_valid, y_valid))
print("score_test:", model.score(X_test, y_test))
