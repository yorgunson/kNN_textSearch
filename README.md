# kNN_textSearch
This repository contains the main Python class and examples of a text search algorithm using k nearest neighbours machine learning algorithm.

A plain explanation of the algorithms and discussion of some results can be found in: http://cforus.blogspot.com/2018/05/a-text-search-method-using-similarities.html

Python codes included in the repository:

- TextSearch_MainClass.py = The Class containing all the methods that computes k-Nearest Neighbors similarity as well as the text handling (tokenization, vectorization, tf-idf calculation). It also contains methods to save/load data and Pandas dataframe handling.

Two examples using the above main class are included:

- AMS_Similarity.py = Conducts a text similarity search among the American Metrorological Society (AMS) scientific article abstracts.

- BBCrecipes_Similarity.py = Conducts a text similarity search among the food recipes from online BBC recipes. 
