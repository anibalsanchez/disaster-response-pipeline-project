import sys
import os
import pickle
import string

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

seed = 42
tokens_to_filter = {'ha', 'u', 'wa'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))
sys.path.append(MODELS_DIR)

class CustomModel:
    @staticmethod
    def tokenize(text):
        lemmatizer = WordNetLemmatizer()
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.strip() and not token.isdigit()]
        lemmatized_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in tokens_to_filter
        ]
        return lemmatized_tokens

    @staticmethod
    def load_model():
        model_path = os.path.abspath(os.path.join(MODELS_DIR, "classifier.pkl"))
        with open(model_path, 'rb') as file:
            model = pickle.load(file, fix_imports = True)
        return model

    @staticmethod
    def build_pipeline():
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                tokenizer=CustomModel.tokenize,
                lowercase=False,
                stop_words='english',
                max_df = 0.9,
                min_df = 2,
                norm = 'l2',
            )),
            ('clf', MultiOutputClassifier(
                estimator=RandomForestClassifier(
                    random_state=seed,
                )
            ))
        ])