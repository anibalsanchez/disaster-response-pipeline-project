# Disaster Response Pipeline Project

## Introduction

As a Udemy [Data Scientist](https://www.udacity.com/enrollment/nd025) Nanodegree Program student, I'm tasked with solving the **Disaster Response Pipeline Project** and publishing the results.

_This project aims to revolutionize disaster response by developing an intelligent system that rapidly categorizes and routes incoming messages to appropriate relief agencies. Using advanced NLP and machine learning, it provides instant multi-category classification through a user-friendly web interface, enabling swift and efficient resource allocation. The goal is to improve disaster management effectiveness significantly, ultimately saving more lives and minimizing crisis impact through data-driven response strategies_

This project applies data engineering skills to analyze disaster data from Appen and build a model for an API that classifies disaster messages. The main components include:

- **ETL Pipeline**: Processes and cleans disaster message data, storing it in a SQLite database. (data/process_data.py)
- **ML Pipeline**: Develops a machine learning model to categorize disaster messages (models/train_classifier.py).
- **Flask Web App**: Provides an interface for emergency workers to input new messages and receive classification results (app/run.py).

Key features:

- Real-time classification of disaster messages
- Data visualizations of the disaster response data
- Utilizes NLP and machine learning techniques

The project showcases:

- Data pipeline development
- Machine learning model creation
- Web application deployment
- Clean, organized code structure

The notebook and source code are available here:

- Blog post: <https://blog.anibalhsanchez.com/en/blogging/...>
- Repository: <https://github.com/anibalsanchez/disaster-response-pipeline-project>

## Quick Start Guide - Setup and Execution

**Prepare the Environment**

- Clone the repository

**Set up Database and Model**

Run these commands in the project's root directory:

- ETL Pipeline (data cleaning and database storage):
  ```
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db > process_data.log
  ```
- ML Pipeline (model training and saving):
  ```
  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl > train_classifier.log
  ```

**Launch the Web Application**

- Navigate to the app directory: `cd app`
- Start the Flask server: `python run.py`

**Access the Web Interface**

- Open your browser and go to: `http://localhost:3001`

Ensure all file paths are correct before running scripts. The web app may take a moment to load due to data processing.

## Exploring the Data

To start the project, I imported the disaster messages and categories and proceeded with the ETL and cleanup.

The ETL (Extract, Transform, Load) process described in the code can be explained in simpler terms as follows:

### Data Loading

The process begins by reading two separate CSV files: one containing messages and another with categories. These two datasets are merged based on a common identifier, creating a comprehensive dataset.

### Data Cleaning

The cleaning process starts by removing duplicate messages and filtering out entries that begin with 'NOTES:' (these messages are manually identified as not related to disasters). The text of each message is then preprocessed by removing URLs and special characters and converting everything to lowercase for consistency.

### Category Handling

The 'categories' column initially contains multiple categories in a single string and is split into separate columns. Each category is converted from a text label to a binary value, making it easier to work with in data analysis or machine learning tasks.

### Data Restructuring

Since the classification is based on the English language, the 'original' message column is eliminated. The newly created category columns are added to the primary dataset.

As mentioned in the project definition, the dataset is imbalanced. There are categories with few samples—for instance, the `water` column. The small number of samples of these categories cannot be used to train the model. To overcome the issue, I defined a threshold (based on the `water` column) of 0.07 to remove the columns in which the ratio is lower. Finally, The process concludes by keeping only the most relevant columns and removing any rows where all category values are zero, ensuring that each remaining entry has at least some categorization.

### Final Cleanup

This ETL process essentially transforms the data from two sources into a clean, structured format ready for further analysis or modeling.

## Feature set exploration

This is the list of categories with meaningful data:

| Name |
|------|
| request |
| aid_related |
| medical_help |
| child_alone |
| food |
| shelter |
| other_aid |
| weather_related |
| floods |
| storm |
| earthquake |
| direct_report |

## Disaster Response Classifier

The Disaster Response Classifier implements a machine-learning process for classifying disaster-related messages.

The text processing processes the text by breaking it into individual words and simplifying them to their base form, **lemmatization** via WordNet Lemmatizer.

The model is built using the **TF-IDF vectorization** and the **Random Forest** method to learn how to simultaneously categorize messages into multiple disaster-related categories (via the Multi-Output Classifier). To compensate the possible imbalance in the dataset, the **balanced class weight** has been activated. The parameter can help improve the model's performance on the minority classes without the need for additional data preprocessing or sampling techniques. The **Grid Search** technique optimizes the model and finds the best parameters.

Finally, the data is split into training and testing sets for evaluation. The confusion matrix and the f1 scores are produced to analyze the results.

The optimized model is saved for later use in the web app.

## Findings in the Disaster Response Dataset

The model performs well across most categories, with accuracies ranging from 0.79 to 1.00. This indicates that the classifier is effective in categorizing disaster-related messages.

| Category | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|----------|
| request | 0.86 | 0.77 | 0.75 | 0.86 |
| aid_related | 0.84 | 0.88 | 0.91 | 0.84 |
| medical_help | 0.90 | 0.77 | 0.50 | 0.89 |
| child_alone | 1.00 | 0.00 | 0.00 | 1.00 |
| food | 0.96 | 0.94 | 0.83 | 0.96 |
| shelter | 0.92 | 0.84 | 0.66 | 0.92 |
| other_aid | 0.79 | 0.57 | 0.33 | 0.77 |
| weather_related | 0.88 | 0.91 | 0.86 | 0.88 |
| floods | 0.93 | 0.91 | 0.61 | 0.92 |
| storm | 0.95 | 0.87 | 0.81 | 0.95 |
| earthquake | 0.96 | 0.94 | 0.83 | 0.96 |
| direct_report | 0.81 | 0.71 | 0.73 | 0.81 |

These are **High-Performing Categories**:

- **Food, Earthquake, and Storm**: These categories show excellent performance with F1-scores above 0.95, indicating high accuracy in identifying these specific disaster-related messages.

These are the **Moderate-Performing Categories**:

- **Request, Aid Related, and Weather Related**: These categories show good performance with F1-scores around 0.85, suggesting reliable classification for these broader categories.

These are the **Lower-Performing Categories**:

- **Other Aid**: This category has the lowest F1-score (0.77), indicating difficulty in accurately classifying messages related to miscellaneous aid requests.
- **Medical Help**: Despite high accuracy (0.90), it has a low recall (0.50) for positive cases, suggesting _the model might be missing many true medical help requests_.

### Imbalanced Data Considerations

Several categories (e.g., Medical Help, Floods) show high accuracy but lower recall for the positive class, indicating a potential class imbalance. This suggests the model might be biased towards the majority class in these categories.

## Conclusion

The classifier performs well overall, especially in specific disaster types like earthquakes and storms. However, there's room for improvement in categories like "Other Aid" and in balancing precision and recall for categories with apparent class imbalance. Further analysis and potential model adjustments could enhance performance in these areas.

## LICENSE

Attribution 4.0 International - CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/anibalsanchez/answering-house-prices-questions-based-on-advanced-regression-techniques">Disaster Response Pipeline Project</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://www.linkedin.com/in/anibalsanchez/">Anibal H. Sanchez Perez</a> is licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Creative Commons Attribution 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""></a></p>
