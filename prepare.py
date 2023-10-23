# imported libraries 
import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from markdown import markdown
from bs4 import BeautifulSoup



import pandas as pd



# defined function to accomplish basic clean actions on text data.
def basic_clean(text_data):
    
    if text_data is None:
        return ""  # Handle the case where text_data is None

    text_data = text_data.lower()
        
    text_data = unicodedata.normalize('NFKD', text_data)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')

    text_data = re.sub(r'[^a-z0-9\s]', '', text_data).lower()

    # Use markdown library to clean 
    text_data = markdown(text_data)

    # Remove HTML tags
    soup = BeautifulSoup(text_data, 'html.parser')
    text_data = soup.get_text()
    
    # Return the cleaned data
    return text_data



# defined function to apply tokenizer object onto text dat and return data as str values.
def tokenize(text_data):
    
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    text_data = tokenizer.tokenize(text_data, return_str=True)

    text_data = re.sub(r"[^a-z0-9\s]", "", text_data)

    text_data = re.sub(r"\s\d{1}\s", "", text_data)
    
    return text_data



# defined function used to stem text in data and joins them with spaces as a string value
def stem(text_data):
    
    ps = nltk.porter.PorterStemmer()

    stems = [ps.stem(word) for word in text_data.split()]
    
    text_data_stemmed = ' '.join(stems)
    
    return text_data_stemmed 




# defined function to lemmatize text in data and return the text as a string in a sentence with "lemmas"
def lemmatize(string):

    wnl = nltk.stem.WordNetLemmatizer()

    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    string = ' '.join(lemmas)

    return string



def remove_stopwords(text_data, extra_words=None, exclude_words=None):
    # stopwords list
    stopwords_list = stopwords.words('english')

    # If extra_words are provided, add them to the stopwords_list
    if extra_words:
        stopwords_list.extend(extra_words)

    # If exclude_words are provided, remove them from the stopwords_list
    if exclude_words:
        stopwords_list = [word for word in stopwords_list if word not in exclude_words]

    # Tokenize the text data and remove stopwords
    words = [word for word in text_data.split() if word not in stopwords_list]

    # Join the words back 
    new_text_data = ' '.join(words)

    return new_text_data

# extra_words = ["framework", "with"]
# exclude_words = ["can"]

# result = remove_stopwords(data, extra_words, exclude_words)
# print(result)

STOPWORDS = ['ro']

def clean(text):
    'A simple function to cleanup text data'
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + STOPWORDS
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]



# defined function to accomplish preparation of text data
def prep_text_data(df, column, extra_words=[], exclude_words=[]):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Apply text preparation functions to the specified column
    df_copy['clean_readme'] = df_copy[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords,
                                  extra_words=extra_words,
                                  exclude_words=exclude_words)
    
    df_copy['stemmed'] = df_copy['clean_readme'].apply(stem)
    df_copy['lemmatized'] = df_copy['clean_readme'].apply(lemmatize)
    
    return df_copy

# prep_text_data(news_df, 'original', extra_words = ['ha'], exclude_words = ['no']).head()