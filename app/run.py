import sys
import os
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))
sys.path.append(MODELS_DIR)

from custom_model import CustomModel

app = Flask(__name__)

message_pattern = r'[^a-zA-Z\s]'
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

fully_cleaned_columns = ['request', 'aid_related', 'medical_help', 'food', 'shelter',
                        'other_aid', 'weather_related', 'floods', 'storm', 'earthquake',
                        'direct_report']
all_category_columns = ['related', 'request', 'offer', 'aid_related', 'medical_help',
                        'medical_products', 'search_and_rescue', 'security', 'military',
                        'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
                        'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
                        'transport', 'buildings', 'electricity', 'tools', 'hospitals',
                        'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
                        'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']

def load_df_from_db():
    db_path = os.path.abspath(os.path.join(BASE_DIR, "../data", "DisasterResponse.db"))
    engine = create_engine('sqlite:///'+db_path)
    return pd.read_sql_table('messages', engine)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    df = load_df_from_db()

    # extract data needed for visuals
    Y = df.drop(['id', 'message', 'genre'], axis=1)
    total = len(Y)
    all_category_columns = Y.columns
    category_counts = df[all_category_columns].sum()

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=all_category_columns,
                    y=category_counts.div(total) * 100
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts.div(total) * 100
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    model = CustomModel.load_model()
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(all_category_columns, classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=False)


if __name__ == '__main__':
    main()