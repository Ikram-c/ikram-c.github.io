# Data science Portfolio

------------------------
# IBM Machine Learning Certification Projects

## IBM ML Regression (Weather Prediction project)
(https://github.com/Ikram-c/IBM-ML-Regression-Weather-Prediction-project-)

The aim of the project was to find an optimized model for weather prediction using the szeged-weather dataset from Kaggle. Data was imported from a CSV into a Pandas dataframe and EDA was conducted to find nulls, the distribution (using a histogram) and plotting a Seaborn correlation Matrix. 

The categorical data was encoded using Sklearn's LabelEncoder and a test train split was used to split the data into training and testing sets.

The following Regression models were used (models were tested with RMSE and r^2 for accuracy):
- Simple Linear regression
- Polynomial Regression (optimised using Kfolds with 10 splits)
- LASSO Regression
- Polynomial Ridge Regression
![Regression plot of weather](https://user-images.githubusercontent.com/68299933/215764240-a61e96c5-fcba-406f-9ab9-ae86e175e4df.jpg)

(Sample of regression plot)


-------------------------------------------------------
## Mushroom Supervised Classification Project 
(https://github.com/Ikram-c/mushroom_classification_IBM)

The aim of the project was to use supervised classification models to predict whether a mushroom was poisonous to eat or not (***Binary classification***). The data was sourced from the Mushroom Classification dataset on kaggle (https://www.kaggle.com/datasets/uciml/mushroom-classification)

Data was imported from a CSV into a Pandas dataframe and EDA was conducted to find nulls, the distribution (using a histogram) and plotting a Seaborn correlation Matrix. *** Cardinality*** was also checked to help assess which features to use for the model.

The categorical data was encoded using Sklearn's LabelEncoder and a test train split was used to split the data into training and testing sets.

The following Supervised classification models were then used (it should be noted that more models were planned to be used and will be in future updates):

- ***Logistic Regression***
- ***KNeighborsClassifier***
- ***DecisionTreeClassifier***

The models were assessed using Confusion matrices and F1 scores.
![Classification plot confusion matrix](https://user-images.githubusercontent.com/68299933/215765549-d1a99592-e2dc-4579-acb6-9896819eda67.jpg)

(Sample of confusion matrix plot of logistic regression classification)



------------------------------------------------------------------------------

## Mall Customer Segmentation Unsupervised Clustering Project
(https://github.com/Ikram-c/mall_customer_segmentation_unsupervised_clustering_IBM)

The aim of this project was to look for ways to ***cluster*** two segments of data together for a hypotehtical business to make better business decisions. The data was sourced from the "Mall Customer Segmentation Data" dataset from kaggle (https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

Outliers were managed using two different methods: the first was to check for datapoints that are more than three standard deviations from the mean and then flagging the values (this did not yield usuable results). The second method used the upper and lower boundary technique (which fixed the outlier).

A scatter plot was produced to see if clusters could be visually identified before using any machine learning algorithms and 5 distinct clusters were observed.

The following Unsupervised classification models were then used:
- ***KMeans***
- ***Meanshift***
- ***DBScan***

***PCA*** was then conducted to try to improve the performance of the algorithms and the same models were then ran again. Models were assessed through their own scatterplots as well as their Silhouette score.
![image](https://user-images.githubusercontent.com/68299933/215767979-57f65ec0-dcf4-4e8b-a1a7-4dd50d926eb1.png)

(sample of Kmeans plot with PCA)


-------------------------------------------------------------------

## Credit Card Fraud Detection Deep Learning (part of IBM ML certification)

The aim of the project was to use Deep Learning, specifically autoencoding, to help classify which credit card transactions are fraudulent. The data used for this project is the "Credit Card Fraud Detection" dataset. The dataset was imbalanced with only 492 fraud transactions occuring out of 284,807 total transactions thus requiring Deep learning.

For the EDA process the data was checked for nulls and its distribution was checked through the ***Anderson-Darling*** test, the ***Shapiro-Wilk*** test and the ***D'Agostino's K^2*** test and normalization tests were run using the ***MinMaxScaler***, ***StandardScaler*** scalers and using ***log_10*** to try to normalize the data. A ***t-SNE*** plot was used to vizualise data using a sample of 2000 non fraudulent cases.

The hyperparameters of the autoencoder were modified to try to improve the accuracy of the model resulting in effectively 8 different deep learning models which were used in conjunction with 3 supervised classification models. 
The hyperparmeters for the autoencoders that were used were:
- ***MSE loss function***
- ***Binary crossentropy loss function***
- ***SGD optimizer***
- ***Adam optimizer***
- ***sigmoid activation function***
- ***tanh activation function***

These hyperparemeters were used in various combinations with each other and were then used with the following supervised clustering algorithms:
- ***Logistic Regression***
- ***KNeighbors Classifier***
- ***Decision Tree Classifier***
The performance of the models were then assessed using their F1 score

![sample t-SNE plot](https://user-images.githubusercontent.com/68299933/215770115-9819bab2-3f96-463d-8107-1d3fdaaf26a7.jpg)

(sample of t-SNE plot)

-------------------------------------------------------------------

## Course Recommender System

The aim of this project was to use various machine learning models for a course recommender system. The dataset which was used was the IBM Course Recommendations dataset which is a collection of data related to IBM courses (specifically those on python) and the interactions students had with them (these were saved as CSVs).
As part of the EDA a ***word cloud** was also used.

For this project both supervised and unsupervised models were used which were then compared with each other. The project itself consists of several notebooks (listed below) along with a pdf of a summary of the project.

The machine learning models used for this project included:
##### Unsuperivsed Learning:
- ***Using dot product to compare vectors for recommendations***
- ***Using Bag of Words (Bows) and a similarity matrix***
- ***Clustering and PCA***

#### Supervised Learning:
- ***KNN from surprise library***
- ***NMF from surprise library***
- ***Tensor flow Neural Network classifier using embeddings***

![comparison of supervised models](https://user-images.githubusercontent.com/68299933/215771976-d9d1eef3-7f5e-4117-a91e-7c09d0fa71b8.jpg)

(sample of comparison of supervised models)


----------------------------------------------
# Other Portfolio Projects

## Bjj classification and Extraction (Sports Data Web scrape and Supervised ML classification)
(https://github.com/Ikram-c/bjj_classification_and_extraction)

The aim of the project was to generate an automated system to analyze Brazilian Jiu-Jitsu (Bjj) data and make predictions about a player’s performance using supervised classification models. Data was ***webscraped*** from a reputable source for Bjj (https://www.bjjheroes.com) which contains an online database of competitors and their competition performance.

The notebook was designed to be used by someone who has very little/ no understanding of coding which is why a series of user inputs are given.

For the web scraping ***Selenium*** and ***bs4*** was used.
For Data frame creation and manipulation ***pandas*** and ***numpy*** were used.
The following ML models were used for classification (***Sklearn***):
- ***Logistic Regression***
- ***Support Vector Machine***
- ***Random Forest Classifier***
- ***KNN***
- ***Naive bayes***
- ***Decision Tree Classifier***

For assessing the accuracy of the models the following metrics were used (***Sklearn***):
- confusion matrix
- f1 score

![Fight history sample image](https://user-images.githubusercontent.com/68299933/214124410-98832400-23ed-466a-b826-11c450dd7ccb.jpg)

(Sample of fight history)

----------------------------------------------------
## Pyspark basics project
(https://github.com/Ikram-c/Pyspark-Basics)

This project goes over an introduction to pyspark and covers PysparkSQL, Data wrangling and Pyspark ML. 
The data used for the notebook was the california housing prices dataset (https://www.kaggle.com/datasets/camnugent/california-housing-prices). Data preprocessing is also used to ensure that the ML models had some level of optimisation, the ML model used was Elastic net.
![example of summary statistics pyspark](https://user-images.githubusercontent.com/68299933/215772988-54c650c4-cf2b-41aa-82f9-0f503867e475.jpg)

(sample of summary statistics pyspark)

-------------------------------------------------------------
## Selenium typing game project
(https://github.com/Ikram-c/Selenium_proj2_typegame)

The aim of this project was to demonstrate how selenium and beautifulsoup (***webscraping libraries***) can be used to complete a basic online game using some of its simplest funcionalities. source website of game: https://www.typingstudy.com/. 

The game was intialized the game by browsing the webpage of the source website of the game. The required inputs were then read (annotation 1 in sample image of typewriter game) and then sent the corresponding input as keyboard inputs (in annotation 2 of sample image of typewriter game). 
![sample image of typewriter game](https://user-images.githubusercontent.com/68299933/215791939-181c4694-33a7-43bb-ae9f-6988902940f9.jpg)

(sample image of typewriter game)
-----------------------------------------------------------------------------------
## Seaborn Mini Project
( https://github.com/Ikram-c/Seaborn-Mini-Project)

The aim of this project was to cover the basics of Seaborn. The following features were looked at:
- Distribution Plots
- Categorical Data Plots
- Matrix Plots
- Grids
- Regression Plots
- Plot Customization

![Sample of testing seaborn lmplot()](https://user-images.githubusercontent.com/68299933/216019762-057d64f3-78e3-4d70-8750-240cb6b55c24.jpg)

(Sample of testing seaborn lmplot())

------------------------------------------------------------------------------------------
## Matplotlib mini project
(https://github.com/Ikram-c/MatplotLib-mini-project)
The aim of this project was to cover the basics of Matplotlib. The following features were looked at:
- Importing Matplotlib
- The basics
- Matplotlib Object Oriented Method
- Figure Size, Aspect Ratio and DPI
- Legends, Labels and Titles
- Setting Colors, Linewidths, Linetypes
- Plot Range
- Special Plot Types

![Sample of testing matplotlib line and marker styles](https://user-images.githubusercontent.com/68299933/216019834-264949cb-c039-408b-95d1-702667616037.jpg)

(Sample of testing matplotlib line and marker styles)

------------------------------------------------------------------------------------------
## EDA using Numpy and Pandas
(https://github.com/Ikram-c/EDA-using-Pandas-and-Numpy)

The aim of this project was to use Pandas and Numpy for EDA for the Microsoft adventureWorks cycles dataset (https://www.kaggle.com/jahias/microsoft-adventure-works-cycles-customer-data).

The main goals of the the EDA process is to:
- Identify and handle missing values with Pandas.
- Implement groupby statements for specific segmented analysis.
- Use apply functions to clean data with Pandas.
The following functions were covered:
- Finding Missing Data
- GroupBy functions
- Apply funcions

![Sample of checking stats of dataset](https://user-images.githubusercontent.com/68299933/216018675-0fb3968e-88ab-4869-9937-3a7f6d28185e.jpg)

(Sample of checking stats of dataset)

----------------------------------------------------------
## Number Guessing Game
(https://github.com/Ikram-c/Number-guessing-game)

The aim of the project was to create a Number guessing game in which you generate a random number and compare it to a user's input. This project used globals and randint

Requirements:
- Tell the user the number range they will be guessing within
- Ask the user for an input
- Print what they answered and state if it is correct or incorrect
- Then print whether or not it is higher or lower
- If it is incorrect, allow user to guess a total of 3 times
- If they are out of guesses or have guessed the correct number Print out their guesses and the correct number
- Ask the user if they want to play again
- If they decide to stop playing, output the number of attempts and correct guesses

![number guessing game flow chart](https://user-images.githubusercontent.com/68299933/216026817-ab6718b9-20cf-49c5-9667-958b20950e35.jpg)

(number guessing game flow chart)
