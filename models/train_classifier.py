import sys
import numpy as np
import pandas as pd
import pickle
import warnings

from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from custom_model import CustomModel

warnings.filterwarnings("ignore", category=UserWarning)

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df =  pd.read_sql_table('messages', engine)
    Y = df.drop(['id', 'message', 'genre'], axis=1)

    return df['message'], Y, Y.columns

def build_model():
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_features': [1000, 2000, 4000],
        'clf__estimator__n_estimators': [100, 200, 300, 500],
        'clf__estimator__max_depth': [10, 20],
        'clf__estimator__class_weight': ['balanced', 'balanced_subsample'],
    }

    return GridSearchCV(
        CustomModel.build_pipeline(),
        parameters,
        n_jobs=-1,
        verbose=2
    )

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    for i, category in enumerate(category_names):
        print(f"\n# Evaluation for {category}")
        accuracy = accuracy_score(Y_test.iloc[:, i], Y_pred[:, i])
        precision = precision_score(Y_test.iloc[:, i], Y_pred[:, i])
        recall = recall_score(Y_test.iloc[:, i], Y_pred[:, i])
        f1 = f1_score(Y_test.iloc[:, i], Y_pred[:, i], average='weighted')

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision score: {precision:.2f}")
        print(f"Recall score: {recall:.2f}")
        print(f"F1-score: {f1:.2f}")

    macro_f1 = np.mean([f1_score(Y_test.iloc[:, i], Y_pred[:, i], average='weighted') for i in range(len(category_names))])
    print(f"\nMacro-averaged F1-score: {macro_f1:.2f}")

    return

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    pass

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        start_time = datetime.now()
        print(f"Process started at: {start_time}")

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        # model = build_pipeline()
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

        end_time = datetime.now()
        print(f"Process ended at: {end_time}")
        print(f"Total execution time: {end_time - start_time}")


    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()