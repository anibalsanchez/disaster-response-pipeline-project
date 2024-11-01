import sys
import pandas as pd
import re
import numpy as np

from IPython.display import display
from sqlalchemy import create_engine

head_n_of_records = 5
pd.options.display.max_columns = None
pd.options.display.max_rows = None

url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
message_pattern = r"[^a-zA-Z]"

almost_empty_threshold = 0.07
almost_empty_categories = ['aid_centers', 'buildings', 'child_alone', 'clothing', 'cold', 'death', 'electricity', 'fire',
                           'hospitals', 'infrastructure_related', 'medical_products', 'military',
                           'missing_people', 'money', 'offer', 'other_infrastructure', 'other_weather',
                           'refugees', 'related', 'search_and_rescue', 'security', 'shops', 'tools',
                           'transport', 'water']

def load_data(messages_filepath, categories_filepath):
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    return pd.merge(df_messages, df_categories, on='id')

def remove_rows_with_all_zeros(df):
    Y = df.drop(['id', 'message', 'genre'], axis=1)
    final_columns = Y.columns
    df = df.loc[~(df[final_columns] == 0).all(axis=1)]

    return df

def clean_data_base(df):
    df = df.drop_duplicates(subset='message', keep='first')
    df = df.reset_index(drop=True)

    df = df[~df['message'].str.startswith('NOTES:')]

    df['message'] = df['message'].str.replace(url_pattern, 'urlplaceholder', regex=True)
    df['message'] = df['message'].str.replace(message_pattern, ' ', regex=True)
    df['message'] = df['message'].str.lower()

    categories_split = df['categories'].str.split(pat=';', expand=True)
    column_names = categories_split.iloc[0].apply(lambda x: x.split('-')[0])
    categories_split.columns = column_names

    display(categories_split.columns)

    for column in categories_split.columns:
        categories_split[column] = categories_split[column].str.split('-').str[1]
        categories_split[column] = categories_split[column].astype(int)

    df.drop('original', axis=1, inplace=True)

    df = pd.concat([df.drop('categories', axis=1), categories_split], axis=1)
    df = remove_rows_with_all_zeros(df)

    return df

def clean_data_full(df):
    df = clean_data_base(df)

    ## Not enough samples, removing rows and columns
    for almost_empty_category in almost_empty_categories:
        df.drop(almost_empty_category, axis=1, inplace=True)

    df = remove_rows_with_all_zeros(df)

    return df

def show_value_counts(df):
    total_rows = len(df)
    for column in df.columns:
        if column == 'message':
            continue

        if column == 'id':
            continue

        print(f"\nValue counts for column '{column}':")

        value_ratios = df[column].value_counts().div(total_rows)
        print(value_ratios)

        if np.any(value_ratios < almost_empty_threshold):
            print(f"\n    The column '{column}' data ratio is less than 0.07.")

        print("-" * 40)

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', con = engine, if_exists='replace', index=False)

    return df

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        # df = clean_data_full(df)

        # Due to the exercise scope, we are going to use only the clean_data_base function
        # to clean the data and keep the 36 categories
        df = clean_data_base(df)

        show_value_counts(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()