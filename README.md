<!--Created Anchor links to navigate read me better-->

- [Project Description](#project-description)
- [Project Goal](#project-goal)
- [Initial Thoughts](#initial-thoughts)
- [Plan](#the-plan)
- [Data Dictionary](#data-dictionary)
- [Steps to Reproduce](#steps-to-reproduce) 
- [Conclusion](#conclusion)
	- [Takeaway and Key Findings](#takeaways-and-key-findings)
	- [Recommendations and Next Steps](#recommendations-and-next-steps)


----------------------------------

# **Project: NLP-Based GitHub Language Predictor**

![image](https://github.com/Marc-Aradillas/github_nlp_project/assets/106922826/851747ae-b2dd-464c-92e1-626060d7563d)

#### The core aim of this project is to automate the process of identifying the primary programming language used in a repository. We achieve this by implementing a machine learning model that analyzes the README text.

### Project Description

This project is designed to automatically identify the primary programming language used in a GitHub repository by analyzing the text in its README file. By harnessing the power of Natural Language Processing (NLP) and machine learning, we aim to make it easier for users to understand the technology stack of a project.

### Project Goal

1. A project utilizing natural language processing techniques that involve webscraping github repos and return text data to analyze programming language uses.

2. Develop a machine learning model model that can predict the main programming language of a repository, given the text of the README file.

### Initial Thoughts

My initial hypothesis is that text data that are associated with programming, tools, and possibly ide tools may be good text string to identify the programming language of a repo.

## The Plan

* Acquire historical repos names, languages, and readme text data from the GitHub API.
* Prepare data using Regex and the BeautifulSoup Library. 
* Explore data in search of which words, bigramsm and trigrams are usefull.
  * Answer the following initial questions
	* Does categorizing langauges by Python? 
  * ?
 	* ?  
 	* ?
* Develop a Model to predict repository main programming language
  * Use text data identified in explore to help build predictive models of different types
  * Evaluate models on train and validate data
  * Select the best model based on $RMSE$ and $R^2$
  * Evaluate the best model on test data
* Draw conclusions

### Data Dictionary

| **Feature**        | **Data Type** | **Definition**                                       |
|--------------------|---------------|-----------------------------------------------------|
| `language`        | strings         | The opening stock price of TSLA on a given date.    |
| `readme_`        | strings         | The highest stock price of TSLA during the day.    |



## Steps to Reproduce

1. Clone this project repository to your local machine.

2. Install project dependencies by running pip install -r requirements.txt in your project directory.

3. Obtain an API key from the Alpha Vantage website.

4. Create a config.py file in your project directory with your API key using the following format:

> GITUB_API = "YOUR_API_KEY"
 
5. Ensure that config.py is added to your .gitignore file to protect your API key.

6. Run the acquire.py script to fetch stock data from the Alpha Vantage API:

> python acquire.py

7. Execute the prepare.py script for data preprocessing and splitting:

> python prepare.py

8. Explore the dataset and answer initial questions using the explore.py script:

> python explore.py

9. Develop machine learning models by running the model.py script:

> python model.py

10. Evaluate the models, select the best-performing one, and draw conclusions based on the results of the model.py script.


# Conclusion

## Takeaways and Key Findings

- 


## Model Improvement
- 

## Recommendations and Next Steps

- I would recommend

  
- Given more time, the following actions could be considered:
  - Gather more data to improve model performance.
  - Revisit the data exploration phase to gain a more comprehensive dataset.
    - 


>  In this NLP project

