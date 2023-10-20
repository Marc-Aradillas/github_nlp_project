# imported libraries 
import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd



# defined function to accomplish basic clean actions on text data.
def basic_clean(text_data):
        
    text_data = unicodedata.normalize('NFKD', text_data)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')

    text_data = re.sub(r'[^a-z0-9\s]', '', text_data).lower()

    return text_data



# defined function to apply tokenizer object onto text dat and return data as str values.
def tokenize(text_data):
    
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    text_data = tokenizer.tokenize(text_data, return_str=True)

    return text_data



# defined function used to stem text in data and joins them with spaces as a string value
def stem(text_data):
    
    ps = nltk.porter.PorterStemmer()

    stems = [ps.stem(word) for word in text_data.split()]
    
    text_data_stemmed = ' '.join(stems)
    
    return text_data_stemmed 




# defined function to lemmatize text in data and return the text as a string in a sentence with "lemmas"
def lemmatize(text_data):

    wnl = nltk.stem.WordNetLemmatizer()

    lemmas = [wnl.lemmatize(word) for word in text_data.split()]
    
    text_data_lemmatized = ' '.join(lemmas)

    return text_data_lemmatized



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


def clean(string, extra_stopwords):    
    words = remove_stopwords((tokenize(basic_clean(string))), extra_stopwords)
    return words


# defined function to accomplish preparation of text data
def prep_text_data(df, column, extra_words=[], exclude_words=[]):
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords,
                                  extra_words=extra_words,
                                  exclude_words=exclude_words)
    
    df['stemmed'] = df['clean'].apply(stem)
    
    df['lemmatized'] = df['clean'].apply(lemmatize)
    
    return df[['repo', column,'clean', 'stemmed', 'lemmatized']]

# prep_text_data(news_df, 'original', extra_words = ['ha'], exclude_words = ['no']).head()