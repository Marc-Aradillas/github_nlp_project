# imports
import pandas as pd
import numpy as np
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

from sklearn.metrics import classification_report as class_rep

def baseline():

    # Load and preprocess your data
    repos_df = pd.read_csv('processed_repos.csv', index_col=0)
    repos_df.drop(columns=(['repo', 'bigrams', 'trigrams']))
    repos_df = repos_df.dropna()
    
    X = repos_df.text
    y = repos_df.language
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    train_baseline_acc = y_train.value_counts().max() / y_train.shape[0] * 100    
    val_baseline_acc = y_val.value_counts().max() / y_val.shape[0] * 100

    print(f'\nBaseline Accuracy')
    print(f'==================================================')
    print(f'\n\nTrain baseline accuracy: {round(train_baseline_acc)}%\n')
    print(f'\nValidation baseline accuracy: {round(val_baseline_acc)}%\n')


def model_1():
    # Load and preprocess your data
    repos_df = pd.read_csv('processed_repos.csv', index_col=0)
    repos_df.drop(columns=(['repo', 'bigrams', 'trigrams']))
    repos_df = repos_df.dropna()

    X = repos_df.text
    y = repos_df.language

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create TF-IDF vectors
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    # Train a logistic regression model
    lm = LogisticRegression(
        penalty='l2',
        C=1.0,
        fit_intercept=False,
        class_weight='balanced',
        solver='liblinear',
        max_iter=100,
        random_state=42
    )
    lm.fit(X_train_tfidf, y_train)

    # Calculate accuracy scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': lm.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': lm.predict(X_val_tfidf)})
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])

    print(f'\nLogisitic Regression Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    


def model_2():
    # Load and preprocess your data
    repos_df = pd.read_csv('processed_repos.csv', index_col=0)
    repos_df.drop(columns=(['repo', 'bigrams', 'trigrams']))
    repos_df = repos_df.dropna()

    X = repos_df.text
    y = repos_df.language

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create TF-IDF vectors
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    # Train KNN Model
    knn = KNeighborsClassifier(
    n_neighbors=2,  
    weights='distance',  # distance
    p=2,  # Euclidean distance
    algorithm='auto',  # 'ball_tree', 'kd_tree', or 'brute'
    leaf_size=30,  
    metric='euclidean'  # You can choose other metrics or provide custom ones
    )
    knn.fit(X_train_tfidf, y_train)

    # Calculate accuracy scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': knn.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': knn.predict(X_val_tfidf)})
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])

    print(f'\nKNearest Neighbors (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    


def model_3():

    # Load and preprocess your data
    repos_df = pd.read_csv('processed_repos.csv', index_col=0)
    repos_df.drop(columns=(['repo', 'bigrams', 'trigrams']))
    repos_df = repos_df.dropna()
    
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    
    # Encode the target labels
    y_encoded = label_encoder.fit_transform(repos_df.language)
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(repos_df.text, y_encoded, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize and fit the TfidfVectorizer on the training data
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Create the XGBoost classifier instance
    bst = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.25, objective='multi:softprob', num_class=len(label_encoder.classes_))
    
    # Fit the XGBoost model on the training data
    bst.fit(X_train_tfidf, y_train)
    
    # Predict the classes on the validation data
    preds = bst.predict(X_val_tfidf)
    
    # If you want to decode the predicted labels back to their original class names:
    preds_decoded = label_encoder.inverse_transform(preds)

    # Calculate accuracy scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': bst.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': bst.predict(X_val_tfidf)})
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])

    print(f'\nXGBClassifier Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')


def model_4():
    # Load and preprocess your data
    repos_df = pd.read_csv('processed_repos.csv', index_col=0)
    repos_df.drop(columns=(['repo', 'bigrams', 'trigrams']))
    repos_df = repos_df.dropna()

    X = repos_df.text
    y = repos_df.language

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create TF-IDF vectors
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    lm = LogisticRegression(
    penalty='l2',  # L2 regularization (Ridge)
    C=1.0,  # Inverse of regularization strength
    fit_intercept=False,  # Include an intercept
    class_weight='balanced',  # You can set class weights if needed
    solver='liblinear',  # Choose a solver appropriate for your data
    max_iter=100,  # You may need to increase this if the model doesn't converge
    random_state=42  # For reproducibility
    )
    
    lm.fit(X_train_tfidf, y_train)

    # Calculate accuracy scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': lm.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': lm.predict(X_val_tfidf)})
    y_test_res = pd.DataFrame({'actual': y_test, 'preds': lm.predict(X_test_tfidf)})
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])
    test_accuracy = accuracy_score(y_test_res['actual'], y_test_res['preds'])

    print(f'\nFinal Model Logisitic Regression with Hyperparameter tuning')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    print(f'\nTest Accuracy: {test_accuracy:.2f}\n')
    




def johns_model_lr_1():
    # Load and preprocess your data
    train = pd.read_csv('train.csv', index_col=0)
    val = pd.read_csv('val.csv', index_col=0)
    test = pd.read_csv('test.csv', index_col=0)
    
    # repos_df.drop(columns=(['repo', 'bigrams', 'trigrams']))
    train = train.dropna()
    val = val.dropna()
    test = test.dropna()
    
    
    X_train = train.text
    y_train = train.language
    
    X_val = val.text
    y_val = val.language


    # Create TF-IDF vectors
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)


    # Train a logistic regression model
    lm = LogisticRegression(
        penalty='l2',
        C=2.0,
        fit_intercept=True,
        class_weight='balanced',
        solver='liblinear',
        max_iter=8,
        random_state=42
    )
    lm.fit(X_train_tfidf, y_train)

    # Calculate accuracy scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': lm.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': lm.predict(X_val_tfidf)})
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])

    print(f'\nLogisitic Regression Model (Hyperparameters Used)')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')



def johns_model_lr_test():
    # Load and preprocess your data
    train = pd.read_csv('train.csv', index_col=0)
    val = pd.read_csv('val.csv', index_col=0)
    test = pd.read_csv('test.csv', index_col=0)
    
    # repos_df.drop(columns=(['repo', 'bigrams', 'trigrams']))
    train = train.dropna()
    val = val.dropna()
    test = test.dropna()
    
    
    X_train = train.text
    y_train = train.language
    
    X_val = val.text
    y_val = val.language
    
    X_test = test.text
    y_test = test.language
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    # lm = LogisticRegression(
    #     penalty='l2',
    #     C=1.0,
    #     fit_intercept=True,
    #     class_weight='balanced',
    #     solver='liblinear',
    #     max_iter=50,
    #     random_state=42
    # )

    lm = LogisticRegression(
    penalty='l2',  # L2 regularization (Ridge)
    C=2.0,  # Inverse of regularization strength
    fit_intercept=True,  # Include an intercept
    class_weight='balanced',  # You can set class weights if needed
    solver='liblinear',  # Choose a solver appropriate for your data
    max_iter=8,  # You may need to increase this if the model doesn't converge
    random_state=42  # For reproducibility
    )
    
    lm.fit(X_train_tfidf, y_train)

    # Calculate accuracy scores
    y_train_res = pd.DataFrame({'actual': y_train, 'preds': lm.predict(X_train_tfidf)})
    y_val_res = pd.DataFrame({'actual': y_val, 'preds': lm.predict(X_val_tfidf)})
    y_test_res = pd.DataFrame({'actual': y_test, 'preds': lm.predict(X_test_tfidf)})
    train_accuracy = accuracy_score(y_train_res['actual'], y_train_res['preds'])
    val_accuracy = accuracy_score(y_val_res['actual'], y_val_res['preds'])
    test_accuracy = accuracy_score(y_test_res['actual'], y_test_res['preds'])

    print(f'\nFinal Model Logisitic Regression with Hyperparameter tuning')
    print(f'==================================================')
    print(f'\nTrain Accuracy: {train_accuracy:.2f}\n')
    print(f'\nValidation Accuracy: {val_accuracy:.2f}\n')
    print(f'\nTest Accuracy: {test_accuracy:.2f}\n')
    
    
    
def nlp_train_val_test(df, seed = 42):

    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed)
    
    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed)
    
    return train, val, test

from scipy.stats import chi2_contingency

def chi2_test_for_all_words(df):
    """
    Perform a Chi-square test for each word in the dataframe to determine its association with the programming languages.

    Args:
    - df (pd.DataFrame): Dataframe containing word frequencies across programming languages.

    Returns:
    - pd.DataFrame: Dataframe with words and their corresponding p-values.
    """

    # Lists to store results
    words = []
    p_values = []

    # Total word counts for each language
    total_word_counts = df[['C++', 'Python', 'Other']].sum()

    # Iterate over each word in the dataframe
    for index, row in df.iterrows():
        # Construct the contingency table
        contingency_table = [list(row[['C++', 'Python', 'Other']]),
                             list(total_word_counts - row[['C++', 'Python', 'Other']])]
        
        # Perform the Chi-square test
        _, p, _, _ = chi2_contingency(contingency_table)
        
        # Append the results
        words.append(row['word'])
        p_values.append(p)

    # Create a dataframe for the results
    results_df = pd.DataFrame({'word': words, 'p_value': p_values})
    return results_df