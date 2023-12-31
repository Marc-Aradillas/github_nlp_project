import pandas as pd
import nltk
from scipy.stats import chi2_contingency
import visual as v

def counts_and_ratios(df, column):
    """
    Takes in a dataframe and a string of a single column
    Returns a dataframe with absolute value counts and percentage value counts
    """
    labels = pd.concat([df[column].value_counts(),
                    df[column].value_counts(normalize=True)], axis=1)
    labels.columns = ['n', 'percent']
    labels
    return labels

def join_text(df):
    """
    Join text data from a DataFrame based on language labels and combine all text data.

    Args:
        df (pd.DataFrame): The DataFrame containing text data and language labels.

    Returns:
        tuple: A tuple containing the following joined text data:
            - C_repos (str): Concatenated text from the DataFrame where the label is 'C++'.
            - Python_repos (str): Concatenated text from the DataFrame where the label is 'Python'.
            - other_repos (str): Concatenated text from the DataFrame where the label is 'Other'.
            - all_repos (str): Concatenated text from the entire DataFrame.
    """
    # Join all the text from the DataFrame where the label is 'C++'
    cpp_repos = ' '.join(df[df.language == 'C++'].text)

    # Join all the text from the DataFrame where the label is 'Python'
    python_repos = ' '.join(df[df.language == 'Python'].text)

    # Join all the text from the DataFrame where the label is 'Other'
    other_repos = ' '.join(df[df.language == 'Other'].text)

    # Join all the text from the entire DataFrame
    all_repos = ' '.join(df.text)
    
    return cpp_repos, python_repos, other_repos, all_repos

def list_words(df):
    """
    Create lists of words from the 'text' column of a DataFrame based on language labels and for all data.

    Args:
        df (pd.DataFrame): The DataFrame containing 'text' text data and language labels.

    Returns:
        tuple: A tuple containing the following lists of words:
            - cpp_words (pd.Series): Words from the 'text' column for 'C++' labeled repositories.
            - python_words (pd.Series): Words from the 'text' column for 'Python' labeled repositories.
            - other_words (pd.Series): Words from the 'text' column for 'Other' labeled repositories.
            - all_words (pd.Series): Words from the 'text' column for all repositories.
    """
    cpp_words = df[df.language == 'C++'].text.str.split(expand=True).stack()
    python_words = df[df.language == 'Python'].text.str.split(expand=True).stack()
    other_words = df[df.language == 'Other'].text.str.split(expand=True).stack()
    all_words = df.text.str.split(expand=True).stack()
    
    return cpp_words, python_words, other_words, all_words

def word_freq(df):
    """
    Calculate word frequencies for different language labels and for all data in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'text' text data and language labels.

    Returns:
        tuple: A tuple containing the following word frequency Series:
            - c_freq (pd.Series): Word frequencies for 'C++' labeled repositories.
            - python_freq (pd.Series): Word frequencies for 'Python' labeled repositories.
            - other_freq (pd.Series): Word frequencies for 'Other' labeled repositories.
            - all_freq (pd.Series): Word frequencies for all repositories.
    """
    # Create lists of words based on language labels and for all data
    cpp_words, python_words, other_words, all_words = list_words(df)

    # Calculate word frequencies and sort in descending order
    c_freq = pd.Series(cpp_words).value_counts().sort_values(ascending=False).astype(int)
    python_freq = pd.Series(python_words).value_counts().sort_values(ascending=False).astype(int)
    other_freq = pd.Series(other_words).value_counts().sort_values(ascending=False).astype(int)
    all_freq = pd.Series(all_words).value_counts().sort_values(ascending=False).astype(int)
    
    return c_freq, python_freq, other_freq, all_freq

def freq_to_dataframe(df):
    """
    Convert a DataFrame of word frequency Series into DataFrames with specified column names.

    Args:
        df (pd.DataFrame): A DataFrame with word frequency Series.

    Returns:
        tuple: A tuple containing individual DataFrames for each word frequency Series
               and a combined DataFrame with all the word frequencies.
    """
    c_freq, python_freq, other_freq, all_freq = word_freq(df)

    # Convert Series to DataFrames with specified column names
    cpp_freq_df = c_freq.reset_index().rename(columns={'index': 'C++'})
    python_freq_df = python_freq.reset_index().rename(columns={'index': 'Python'})
    other_freq_df = other_freq.reset_index().rename(columns={'index': 'Other'})
    all_freq_df = all_freq.reset_index().rename(columns={'index': 'All'})

    return cpp_freq_df, python_freq_df, other_freq_df, all_freq_df


def counts_df_concat(df):
    """
    Concatenate multiple DataFrames with word frequency information, separated by empty columns.

    Args:
        df (pd.DataFrame): A DataFrame containing word frequency DataFrames.

    Returns:
        pd.DataFrame: A combined DataFrame with word frequencies separated by empty columns.
    """
    cpp_freq_df, python_freq_df, other_freq_df, all_freq_df = freq_to_dataframe(df)
    
    # Calculate the maximum length among DataFrames
    max_len = max(len(cpp_freq_df), len(python_freq_df), len(other_freq_df), len(all_freq_df))

    # Create an empty DataFrame with the same number of rows as the DataFrame with max rows
    empty_column = pd.DataFrame({' ': [''] * max_len})

    # Concatenate the DataFrames with the empty column in between
    top_freq_df = pd.concat([empty_column, all_freq_df, empty_column, empty_column, cpp_freq_df, empty_column, empty_column, python_freq_df, empty_column, empty_column, other_freq_df], axis=1)

    # Set the index to start at 1
    top_freq_df.index = top_freq_df.index + 1

    return top_freq_df

def word_counts(df, reset_index=True):
    """
    Process and sort word frequency DataFrames.

    Args:
        df (pd.DataFrame): DataFrame containing word frequency DataFrames.
        reset_index (bool, optional): Whether to reset the index and start it at 1. Default is True.

    Returns:
        pd.DataFrame: Sorted and processed word counts DataFrame with "word" column as the first column,
                      and index reset based on the reset_index parameter.
    """
    cpp_freq, python_freq, other_freq, all_freq = word_freq(df)
    
    # Concatenate the DataFrames and set column names
    word_counts = (pd.concat([all_freq, cpp_freq, python_freq, other_freq], axis=1, sort=True)
                    .set_axis(['all', 'C++', 'Python', 'Other'], axis=1)
                    .fillna(0)
                    .apply(lambda s: s.astype(int)))

    # Sort by the 'all' column in descending order
    word_counts = word_counts.sort_values(by='all', ascending=False)

    if reset_index:
        # Move the index into a column named "word"
        word_counts['word'] = word_counts.index

        # Reset the index and start it at 1
        word_counts = word_counts.reset_index(drop=True)
        word_counts.index = word_counts.index + 1
        
        # Reorder the columns with "word" as the first column
        word_counts = word_counts[['word', 'all', 'C++', 'Python', 'Other']]

    return word_counts


def get_top_n_ngrams(series, num_words, top_n, remove_delimiter=False):
    """
    Get the top N n-grams from a Series of text data.

    Parameters:
    series (Series): A Series containing text data.
    num_words (int): The number of words to consider for creating n-grams.
    top_n (int): The number of top n-grams to retrieve.
    remove_delimiter (bool): Whether to remove delimiters from the n-grams. Defaults to False.

    Returns:
    Series: A Series with the top N n-grams and their counts.
    """
    # Create n-grams directly from the input series
    ngrams = list(nltk.ngrams(series, num_words))

    if remove_delimiter:
        # Remove delimiters from n-grams by joining without commas
        ngrams = [tuple(" ".join(word.split(",")) for word in ngram) for ngram in ngrams]

    # Create a Series of n-gram counts and retrieve the top N n-grams
    top_ngrams = (pd.Series(ngrams)
                  .value_counts()
                  .head(top_n))

    # Display n-grams without commas
    top_ngrams.index = [' '.join(ngram) for ngram in top_ngrams.index]

    return top_ngrams


def perform_chi_square_test(dataframe, alpha=0.05):
    """
    Perform the Chi-square test on the provided DataFrame and print the result.

    Args:
    dataframe (pd.DataFrame): A DataFrame containing word frequencies across different categories.
    alpha (float): Significance level for the Chi-square test. Default is 0.05.

    Returns:
    None
    """
    # Extracting the word frequency data, assuming the first column is 'word' and should be excluded
    contingency_data = dataframe.iloc[:, 1:].values

    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_data)

    # Print the results
    print(f"Chi-square Statistic: {chi2}")
    print(f"p-value: {p}")
    # print(f"Degrees of Freedom: {dof}")

    # Decision: compare p-value with alpha
    if p < alpha:
        print("We reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")


def q1_vis_test(df, alpha=0.05):
    word_count = word_counts(df, reset_index=False)
    top_word_count = word_count.head(10)
    v.plot_top_words(word_count, column='all', top_n=10, figsize=(10, 6))
    perform_chi_square_test(top_word_count, alpha)

def q2_chi_test(contingency_table, alpha=0.05):
    """
    Perform the Chi-square test of independence on the provided contingency table.

    Args:
    contingency_table (np.array or list of lists): 2D array with the format shown above.
    alpha (float): Significance level for the test. Default is 0.05.

    Returns:
    None
    """
    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Results
    print(f'Chi-square Statistic: {chi2}')
    print(f'P-value: {p}')
    # print(f'Degrees of Freedom: {dof}')
    # print('Expected Frequencies:')
    print(expected)
    
def q2_cont_table(df):
    
    word_count = word_counts(df, reset_index=True)
    
    # Step 1: Extract counts for "build"
    build_counts = word_count[word_count['word'] == 'build'][['C++', 'Python', 'Other']].iloc[0]

    # Step 2: Calculate total counts
    total_counts = word_count[['C++', 'Python', 'Other']].sum()

    # Step 3: Calculate counts for "does not contain 'build'"
    not_build_counts = total_counts - build_counts

    # Step 4: Structure the contingency table
    contingency_table = pd.DataFrame([build_counts, not_build_counts], index=['contains_build', 'not_contains_build'])
    return contingency_table

def q2_vis_test(df, alpha=0.05):
    word_count = word_counts(df, reset_index=False)
    contingency_table = q2_cont_table(df)
    v.plot_top_words(word_count, column='C++', top_n=10, figsize=(10, 6), title= 'The 10 most common words in C++ repos and their ratios')
    perform_chi_square_test(contingency_table, alpha)

def vis_4(df, title):
    cpp_words, python_words, other_words, all_words = list_words(df)
    top_20_python_bigrams = get_top_n_ngrams(python_words, 2, top_n= 10, remove_delimiter=True)
    v.plot_ngrams(top_20_python_bigrams, top_n=10, figsize=(12, 10), title=title)

def vis_5(df, title):
    cpp_words, python_words, other_words, all_words = list_words(df)
    top_20_cpp_trigrams = get_top_n_ngrams(cpp_words, 3, top_n= 10, remove_delimiter=True)
    v.plot_ngrams(top_20_cpp_trigrams, top_n=10, figsize=(12, 10), title=title)


def vis_1(df):
    # Counts and ratios of language labels in a DataFrame
    counts = counts_and_ratios(df, "language")

    # Visualize the distribution
    v.plot_language_distribution(counts)