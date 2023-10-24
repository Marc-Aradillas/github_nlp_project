import pandas as pd
import unicodedata
import re
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
nltk.download('wordnet')

from markdown import markdown


def basic_clean(data):
    """
    Perform basic text cleaning operations on the input data.

    Args:
        data (str): The input text data to be cleaned.

    Returns:
        str: The cleaned text data after applying the following operations:
             - Convert text to lowercase.
             - Normalize text by removing diacritical marks.
             - Remove characters that are not lowercase letters, numbers, apostrophes, or whitespaces.
             - Clean HTML content and remove HTML tags.
    """
    # Convert the text to lowercase
    data = data.lower()
    
    # Remove URLs using regular expression
    data = re.sub(r"http\S+|www\S+", "", data)
    
    # Normalize the text by removing any diacritical marks
    data = unicodedata.normalize('NFKD', data)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')

    # Use markdown library to clean 
    data = markdown(data)

    # Remove HTML tags
    soup = BeautifulSoup(data, 'html.parser')
    data = soup.get_text()
    
    # # Remove any characters that are not lowercase letters, numbers, apostrophes, or whitespaces
    # data = re.sub(r"[^a-z0-9\s']", "", data)

    # Define a regular expression pattern to remove unwanted characters but preserve "C++"
    pattern = r"[^a-zA-Z0-9\s'\+]|(?<!\w)C\+\+(?!\w)"

    # Use the regular expression pattern to remove unwanted characters
    data = re.sub(pattern, "", data)

    # Return the cleaned data
    return data


def tokenize(data):
    """
    Tokenize the input text data using a tokenizer object and apply additional text processing.

    Args:
        data (str): The input text data to be tokenized.

    Returns:
        str: The tokenized and processed text data after performing the following operations:
             - Tokenization using ToktokTokenizer.
             - Removing characters that are not lowercase letters, numbers, or whitespaces.
             - Removing single-digit numbers.
    """
    # Initialize a tokenizer object
    tokenizer = ToktokTokenizer()

    # Tokenize the input data using the tokenizer object
    data = tokenizer.tokenize(data, return_str=True)

    # Remove characters that are not lowercase letters, numbers, or whitespaces
    data = re.sub(r"[^a-z0-9\s]", "", data)

    # Remove single-digit numbers surrounded by spaces
    data = re.sub(r"\s\d{1}\s", "", data)
    
    # Return the processed data
    return data

def stem(data):
    """
    Apply stemming to the input text data using the Porter Stemmer algorithm.

    Args:
        data (str): The input text data to be stemmed.

    Returns:
        str: The stemmed text data after applying stemming to each word.
    """
    # Create an instance of the PorterStemmer class from the nltk library
    ps = nltk.porter.PorterStemmer()
    
    # Split the input data into a list of words
    words = data.split()
    
    # Apply stemming to each word in the input data
    stems = [ps.stem(word) for word in words]

    # Join the stemmed words into a single string with spaces in between
    stemmed_data = ' '.join(stems)

    # Return the stemmed data
    return stemmed_data

def lemmatize(data):
    """
    Apply lemmatization to the input text data using WordNet Lemmatizer.

    Args:
        data (str): The input text data to be lemmatized.

    Returns:
        str: The lemmatized text data after applying lemmatization to each word.
    """
    # Create an instance of WordNetLemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Split the input data into a list of words
    words = data.split()
    
    # Lemmatize each word in the input data
    lemmas = [wnl.lemmatize(word) for word in words]

    # Join the lemmatized words into a single string
    lemmatized_data = ' '.join(lemmas)

    # Return the lemmatized data
    return lemmatized_data


def remove_stopwords(data, extra_words=[], exclude_words=[]):
    """
    Remove stopwords from the input text data while allowing for additional and exclusionary stopwords.

    Args:
        data (str): The input text data from which stopwords will be removed.
        extra_words (list): Additional stopwords to be considered.
        exclude_words (list): Words to be excluded from the list of stopwords.

    Returns:
        str: The text data with stopwords removed based on the provided lists.
    """
    # Create a list of stopwords in English
    stopwords_list = stopwords.words('english')

    # Extend the stopwords_list with the elements from the extra_words list
    stopwords_list.extend(extra_words)

    # Iterate over each word in the exclude_words list
    for word in exclude_words:
        # Check if the word exists in the stopwords_list
        if word in stopwords_list:
            # Remove the word from the stopwords_list
            stopwords_list.remove(word)

    # Split the input data into individual words and filter out stopwords
    words = [word for word in data.split() if word not in stopwords_list]
    
    # Join the filtered words back into a string
    data = ' '.join(words)
    
    # Return the processed data
    return data



def preprocess_text_column(df, extra_words=[], exclude_words=[], method='stem'):
    """
    Preprocess the 'text' column of a DataFrame by applying text cleaning and processing steps.
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        extra_words (list): Additional words to include in stopwords removal.
        exclude_words (list): Words to exclude from stopwords removal.
        method (str): Text processing method ('stem' or 'lemmatize').

    Returns:
        None: The function modifies the DataFrame 'df' in place.
    """
    # Apply basic cleaning and tokenization to 'text_contents' column
    df['text'] = df['text_contents'].apply(basic_clean).apply(tokenize)
    
    # Drop the 'text_contents' column
    df.drop(columns='text_contents', axis=1, inplace=True)
    
    # # Apply stopwords removal and text processing based on the selected method
    # df['text'] = df['text'].apply(lambda x: remove_stopwords(x, extra_words, exclude_words))
    
    if method == 'stem':
        # Apply stemming to the 'text' column
        df['text'] = df['text'].apply(stem)
    elif method == 'lemmatize':
        # Apply lemmatization to the 'text' column
        df['text'] = df['text'].apply(lemmatize)
    
    # Apply stopwords removal and text processing based on the selected method
    df['text'] = df['text'].apply(lambda x: remove_stopwords(x, extra_words, exclude_words))
    
    return df

def remove_invalid_rows(df):
    """
    Remove rows from a DataFrame where 'language' is None and 'text' is 'failtoloadtext'.
    
    Args:
        df (pd.DataFrame): The DataFrame to remove rows from.

    Returns:
        None: The function modifies the DataFrame 'df' in place.
    """
    # Remove rows where 'language' is None
    df = df[df['language'].notna()]
    
    # Remove rows where 'text' is 'failtoloadtext'
    df = df[df['text'] != 'failtoloadtext']
    
    return df


def remove_non_languages(df, non_languages):
    """
    Remove rows from a DataFrame where the specified column contains any of the given items.

    Args:
        df (pd.DataFrame): The DataFrame to remove rows from.
        non_languages (list): A list of non-language items to check for and remove if found.

    Returns:
        pd.DataFrame: The DataFrame with rows removed based on the specified non-language items.
    """
    return df[~df['language'].isin(non_languages)]

def categorize_language(language, labeled_languages):
    """
    Categorize a given language into one of the specified categories or 'Other' if it doesn't match any.

    Args:
        language (str): The language to be categorized.
        labeled_languages (list): List of acceptable languages for categorization.

    Returns:
        str: The categorized language, which is either one of the acceptable languages or 'Other'.
    """
    if language in labeled_languages:
        return language
    else:
        return 'Other'
    

def add_bigrams(data):
    data.str.split(expand=True).stack()


def process_dataframe(df, extra_words=[], exclude_words=[], method='stem', labeled_languages=[], non_languages=[]):
    """
    Process a DataFrame by applying text preprocessing, filtering, and language categorization operations.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        extra_words (list): Additional words to include in stopwords removal.
        exclude_words (list): Words to exclude from stopwords removal.
        method (str): Text processing method ('stem' or 'lemmatize').
        labeled_languages (list): List of acceptable languages.
        non_languages (list): List of non-language items to be removed from the DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame with rows removed based on non-language items.
    """
    # Preprocess the 'text' column
    df = preprocess_text_column(df, extra_words, exclude_words, method)
    
    # Remove rows with invalid data
    df = remove_invalid_rows(df)
    
    # Remove rows with non-language items
    df = remove_non_languages(df, non_languages)
    
    # Categorize the 'language' column based on labeled_languages
    df['language'] = df['language'].apply(categorize_language, args=(labeled_languages,))
    
    df = df[df['text'].notna() & df['text'].str.contains(' ')]
    
    return df