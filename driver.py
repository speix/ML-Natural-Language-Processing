import pyprind
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import csv
import os
import re

train_path = "./aclImdb/train/"
test_path = "./aclImdb/test/"

# print('Accuracy: %.3f' % classifier.score(x_test, y_test))

only_alnum_pattern = re.compile('([^\s\w]|_)+')
no_tags_pattern = re.compile('<[^>]*>')
stopwords = set(open('stopwords.en.txt', 'r').read().split())


def save(filename, predictions):

    output = open(filename, 'a')

    for prediction in predictions:
        output.write(str(prediction) + "\n")

    output.close()


def prediction_file_size(path):
    with open(path, 'rU') as csv_te:
        reader = csv.reader(csv_te)
        return sum(1 for row in reader)


def tokenizer(text):
    text = no_tags_pattern.sub('', text.lower())
    text = only_alnum_pattern.sub('', text.lower())
    tokens = [w for w in text.split() if w not in stopwords]
    return tokens


def stream(path, scope):

    if scope == 'training':

        with open(path, 'rU') as csv_tr:
            next(csv_tr)
            reader = csv.reader(csv_tr)
            for row in reader:
                text, label = row[0], int(row[1])
                yield text, label

    elif scope == 'predicting':

        with open(path, 'rU') as csv_te:
            next(csv_te)
            reader = csv.reader(csv_te)
            for row in reader:
                text = row[1]
                yield text


def batch(chunk, size, scope):

    if scope == 'training':

        documents, labels = list(), list()
        try:
            for _ in range(size):
                text, label = next(chunk)
                documents.append(text)
                labels.append(label)
        except StopIteration:
            return None, None

        return documents, labels

    elif scope == 'predicting':

        documents = list()
        try:
            for _ in range(size):
                text = next(chunk)
                documents.append(text)
        except StopIteration:
            return None

        return documents


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    labels = {'pos': 1, 'neg': 0}
    df = pd.DataFrame()
    for label in ('pos', 'neg'):
        path = os.path.join(inpath, label)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as myfile:
                text = myfile.read()
            df = df.append([[text, labels[label]]], ignore_index=True)
    df.columns = ['text', 'polarity']
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv(outpath + name, index=False)


def unigram(documents, prediction_documents):
    vectorizer = HashingVectorizer(decode_error='ignore', ngram_range=(1, 1), preprocessor=None, tokenizer=tokenizer, analyzer='word')
    classifier = SGDClassifier(loss='hinge', penalty='l1')

    chunk = stream(path=documents, scope='training')

    pbar = pyprind.ProgBar(10)
    classes = np.array([0, 1])
    for _ in range(10):
        reviews, labels = batch(chunk, size=2500, scope='training')
        reviews = vectorizer.transform(reviews)
        classifier.partial_fit(reviews, labels, classes=classes)
        pbar.update()

    prediction_size = prediction_file_size(prediction_documents) - 1
    test_chunk = stream(path=prediction_documents, scope='predicting')
    test_reviews = batch(test_chunk, size=prediction_size, scope='predicting')
    test_reviews = vectorizer.transform(test_reviews)

    predictions = classifier.predict(test_reviews)

    save('unigram.output.txt', predictions)


def unigramtfidf(documents, prediction_documents):
    vectorizer = TfidfVectorizer(decode_error='ignore', use_idf=True, ngram_range=(1, 1), preprocessor=None, tokenizer=tokenizer, analyzer='word')
    classifier = Pipeline([('vect', vectorizer), ('clf', SGDClassifier(loss='hinge', penalty='l1'))])

    chunk = stream(path=documents, scope='training')

    pbar = pyprind.ProgBar(10)
    for _ in range(10):
        reviews, labels = batch(chunk, size=2500, scope='training')
        classifier.fit(reviews, labels)
        pbar.update()

    prediction_size = prediction_file_size(prediction_documents) - 1
    test_chunk = stream(path=prediction_documents, scope='predicting')
    test_reviews = batch(test_chunk, size=prediction_size, scope='predicting')

    predictions = classifier.predict(test_reviews)

    save('unigramtfidf.output.txt', predictions)


def bigram(documents, prediction_documents):
    vectorizer = HashingVectorizer(decode_error='ignore', ngram_range=(1, 2), preprocessor=None, tokenizer=tokenizer, analyzer='word')
    classifier = SGDClassifier(loss='hinge', penalty='l1')

    chunk = stream(path=documents, scope='training')

    pbar = pyprind.ProgBar(10)
    classes = np.array([0, 1])
    for _ in range(10):
        reviews, labels = batch(chunk, size=2500, scope='training')
        reviews = vectorizer.transform(reviews)
        classifier.partial_fit(reviews, labels, classes=classes)
        pbar.update()

    prediction_size = prediction_file_size(prediction_documents) - 1
    test_chunk = stream(path=prediction_documents, scope='predicting')
    test_reviews = batch(test_chunk, size=prediction_size, scope='predicting')
    test_reviews = vectorizer.transform(test_reviews)

    predictions = classifier.predict(test_reviews)

    save('bigram.output.txt', predictions)


def bigramtfidf(documents, prediction_documents):
    vectorizer = TfidfVectorizer(decode_error='ignore', use_idf=True, ngram_range=(1, 2), preprocessor=None, tokenizer=tokenizer, analyzer='word')
    classifier = Pipeline([('vect', vectorizer), ('clf', SGDClassifier(loss='hinge', penalty='l1'))])

    chunk = stream(path=documents, scope='training')

    pbar = pyprind.ProgBar(10)
    for _ in range(10):
        reviews, labels = batch(chunk, size=2500, scope='training')
        classifier.fit(reviews, labels)
        pbar.update()

    prediction_size = prediction_file_size(prediction_documents) - 1
    test_chunk = stream(path=prediction_documents, scope='predicting')
    test_reviews = batch(test_chunk, size=prediction_size, scope='predicting')

    predictions = classifier.predict(test_reviews)

    save('bigramtfidf.output.txt', predictions)


def main():
    training_file = './imdb_tr.csv'
    prediction_file = test_path + 'imdb_te.csv'

    if not os.path.exists(training_file):
        imdb_data_preprocess(train_path)

    if not os.path.exists('unigram.output.txt'):
        unigram(training_file, prediction_file)

    if not os.path.exists('bigram.output.txt'):
        bigram(training_file, prediction_file)

    if not os.path.exists('unigramtfidf.output.txt'):
        unigramtfidf(training_file, prediction_file)

    if not os.path.exists('bigramtfidf.output.txt'):
        bigramtfidf(training_file, prediction_file)

if __name__ == "__main__":
    main()
