# Machine Learning - Natural Language Processing
Trained a Stochastic Gradient Descent Classifier with different ngram (Tf-Idf) model representations, to classify imdb reviews as being positive or negative. Tf-idf stands for Term Frequency-Inverse Document Frequency, which actually reflects how important a word is to a document in a collection of corpus.

The first step is to merge all the positive and negative reviews with their respective labels (0 for negative reviews with score 1-4 AND 1 for positive reviews with score 7-10). Next we shuffle the set (reindexing) and create a proper training-set.csv which the classifier will use for sentiment analysis. 

For every batch of reviews, the program strips them of html tags, stopwords or any other irrelevant characters and "feeds" them to the classifier. The goal of the classifier is to predict new reviews as either being positive or negative.

### Usage

* Required libraries: sklearn, pandas, numpy, pyprind

```
python driver.py
```

### Testing our SGD Classifier
In a sample of N labeled reviews, we can split the set, using 10% as a testing set and measure the accuracy using different ngram representations like unigram, bigram etc

We can also inject a very powerful sklearn module like GridSearchCV, which uses our estimator with a grid of parameters to determine the best ones to fit our sample.

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

### Results
