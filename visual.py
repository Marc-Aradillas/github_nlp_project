from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import explore as exp

def plot_top_words(word_counts, column='all', top_n=20, figsize=(12, 10), title=None):
    """
    Plots a horizontal stacked bar chart of the top N words in a DataFrame.

    Parameters:
    word_counts (DataFrame): A DataFrame containing word counts and categories.
    column (str): The name of the column to sort by. Defaults to 'all'.
    top_n (int): The number of top records to plot. Defaults to 20.
    figsize (tuple): Width, height in inches. Defaults to (12, 10).
    title (str): The title of the plot. If None, a default title will be generated. Defaults to None.
    """
    # Select the top N rows by specified column
    top_words = word_counts.nlargest(top_n, column)

    # Sort the DataFrame by specified column in descending order for the plot
    top_words = top_words.sort_values(by=column, ascending=True)

    # Create a stacked bar chart with a cleaner style
    ax = top_words[['C++', 'Python', 'Other']].plot(kind='barh', stacked=True, figsize=figsize, width=0.7)

    # Set plot title with larger font size
    if title is None:  # Check if a custom title is provided
        title = f'The {top_n} most common words'
    
    plt.title(title, fontsize=16)

    # Remove the box, x axis, and y tick marks
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks([])  # Remove x ticks
    plt.yticks(fontsize=14)  # Keep y labels but increase font size
    plt.gca().tick_params(left=False)  # Remove y ticks

    # Calculate total width of each bar
    total_widths = top_words['all']
    total_widths.reset_index(drop=True, inplace=True)  # Reset index to ensure correct matching

    num_bars = len(total_widths)
    num_categories = 3  # 'C++', 'Python', 'Other'

    # Add percentage labels inside each section of the bar, with conditional display
    for i, patch in enumerate(ax.patches):
        width, height = patch.get_width(), patch.get_height()
        x, y = patch.get_xy() 
        bar_index = i % num_bars  # Adjusted indexing to match the patch order
        total_width = total_widths.iloc[bar_index]
        section_percentage = (width / total_width) * 100 if total_width > 0 else 0

        if section_percentage >= 10:  # Only show percentage if it's 10% or greater
            ax.text(x + width / 2, 
                    y + height / 2, 
                    f'{int(section_percentage)}%',  # Rounded to the nearest whole number
                    ha='center', 
                    va='center',
                    fontsize=10,  # Adjusted font size for readability
                    color='white')  # Set text color to white for better readability

    # Add total count at the end of each bar
    for i, total_width in enumerate(total_widths):
        ax.text(total_width + 3,  # Slightly offset from the end of the bar
                i,  # Y-coordinate (aligned with each bar)
                str(total_width),  # The total count for the word
                va='center',  # Vertically align in the center of the bar
                fontsize=10)  # Consistent font size

    # Show the plot
    plt.tight_layout()
    plt.show()



def word_clouds(df):
    """
    Generate and display word clouds for different sets of words based on language labels.

    Args:
        df (pd.DataFrame): The DataFrame containing 'readme' text data and language labels.
    """
    # Extract words from the DataFrame using the list_words function
    cpp_words, python_words, other_words, all_words = exp.list_words(df)

    # Generate word clouds for each set of words
    c_cloud = WordCloud(background_color='white', height=1000, width=1000).generate(' '.join(cpp_words))
    python_cloud = WordCloud(background_color='white', height=1000, width=1000).generate(' '.join(python_words))
    other_cloud = WordCloud(background_color='white', height=1000, width=1000).generate(' '.join(other_words))
    all_cloud = WordCloud(background_color='white', height=1000, width=1000).generate(' '.join(all_words))

    # Create subplots for each word cloud
    plt.figure(figsize=(16, 12))

    # Subplot for C++
    plt.subplot(2, 2, 1)
    plt.imshow(c_cloud)
    plt.title('C++ Words')
    plt.axis('off')

    # Subplot for Python
    plt.subplot(2, 2, 2)
    plt.imshow(python_cloud)
    plt.title('Python Words')
    plt.axis('off')

    # Subplot for Other
    plt.subplot(2, 2, 3)
    plt.imshow(other_cloud)
    plt.title('Other Words')
    plt.axis('off')

    # Subplot for All Words
    plt.subplot(2, 2, 4)
    plt.imshow(all_cloud)
    plt.title('All Words')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def calculate_ratios(df, min_count_cpp=100, min_count_python=100, min_count_other=100, sort_column='p_C++'):
    """
    Filter rows where 'C++', 'Python', and 'Other' counts are greater than or equal to the specified minimum counts
    and calculate the percentage ratios for 'C++', 'Python', and 'Other' columns.

    Parameters:
    word_counts (DataFrame): A DataFrame containing word counts for different categories.
    min_count_cpp (int): The minimum count threshold for 'C++' category. Defaults to 100.
    min_count_python (int): The minimum count threshold for 'Python' category. Defaults to 100.
    min_count_other (int): The minimum count threshold for 'Other' category. Defaults to 100.
    sort_column (str): The column to use for sorting. Defaults to 'p_C++'.

    Returns:
    DataFrame: A DataFrame with filtered rows and calculated percentage ratios, sorted by the specified column.
    """
    # The word frequencies will be stored in the 'word_counts' variable
    word_counts = exp.word_counts(df, reset_index=False)
    
    # Filter rows where 'C++', 'Python', and 'Other' counts are greater than or equal to specified minimum counts
    filtered_word_counts = word_counts[
        (word_counts['C++'] >= min_count_cpp) &
        (word_counts['Python'] >= min_count_python) &
        (word_counts['Other'] >= min_count_other)
    ].copy()

    # Calculate the percentage ratios for 'C++', 'Python', and 'Other' columns
    filtered_word_counts['p_C++'] = (
        (filtered_word_counts['C++'] / (filtered_word_counts['C++'] + filtered_word_counts['Python'] + filtered_word_counts['Other'])) * 100
    ).round()
    filtered_word_counts['p_Python'] = (
        (filtered_word_counts['Python'] / (filtered_word_counts['C++'] + filtered_word_counts['Python'] + filtered_word_counts['Other'])) * 100
    ).round()
    filtered_word_counts['p_Other'] = (
        (filtered_word_counts['Other'] / (filtered_word_counts['C++'] + filtered_word_counts['Python'] + filtered_word_counts['Other'])) * 100
    ).round()

    # Sort the DataFrame by the specified column in descending order
    sorted_word_counts = filtered_word_counts.sort_values(by=sort_column, ascending=False)

    # Select the top and bottom rows using the pipe method
    selected_word_counts = sorted_word_counts.pipe(lambda df: pd.concat([df.head(), df.tail()]))

    return selected_word_counts


