import matplotlib.pyplot as plt
import pandas as pd

def plot_top_words(word_counts, column='all', top_n=20, figsize=(12, 10)):
    """
    Plots a horizontal stacked bar chart of the top N words in a DataFrame.

    Parameters:
    word_counts (DataFrame): A DataFrame containing word counts and categories.
    column (str): The name of the column to sort by. Defaults to 'all'.
    top_n (int): The number of top records to plot. Defaults to 20.
    figsize (tuple): Width, height in inches. Defaults to (12, 10).
    """
    # Select the top N rows by specified column
    top_words = word_counts.sort_values(by=column).tail(top_n)

    # Sort the DataFrame by specified column
    top_words = top_words.sort_values(by=column, ascending=True)

    # Create a stacked bar chart with a cleaner style
    ax = top_words[['C++', 'Python', 'Other']].plot(kind='barh', stacked=True, figsize=figsize, width=0.7)

    # Set plot title with larger font size
    plt.title(f'Proportion of C++, Python, and Other for the {top_n} most common words', fontsize=16)

    # Remove the box, x axis, and y tick marks
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xticks([])  # Remove x ticks
    plt.yticks(fontsize=14)  # Keep y labels but increase font size
    plt.gca().tick_params(left=False)  # Remove y ticks

    # Calculate total width of each bar
    total_widths = top_words[['C++', 'Python', 'Other']].sum(axis=1)

    # Add percentage labels inside each section of the bar, with conditional display
    for i, patch in enumerate(ax.patches):
        width, height = patch.get_width(), patch.get_height()
        x, y = patch.get_xy() 
        total_width = total_widths.iloc[i // 3]  # Corrected indexing here
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
        ax.text(total_width + 50,  # Slightly offset from the end of the bar
                i,  # Y-coordinate (aligned with each bar)
                str(total_width),  # The total count for the word
                va='center',  # Vertically align in the center of the bar
                fontsize=10)  # Consistent font size

    # Show the plot
    plt.tight_layout()
    plt.show()
