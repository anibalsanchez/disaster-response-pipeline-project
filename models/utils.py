import sys
import os
import pickle

from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

seed = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))
sys.path.append(MODELS_DIR)

class CustomModel:
    @staticmethod
    def tokenize(text):
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in text.split()]
        return tokens

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
                stop_words='english'
            )),
            ('clf', MultiOutputClassifier(
                estimator=RandomForestClassifier(
                    random_state=seed,
                    class_weight='balanced'
                )
            ))
        ])