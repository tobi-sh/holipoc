About
=====

This repository includes a brief demonstration of categorizing campaign data based into different topics. You can train a model, have a brief look at some basic metrics and use this model later on to classify random text into categories. Please note that this isn't optimized in any manner. It should demonstrate some concepts, workflows and give you the opportunity to test and play around.

Installation
============

Prerequisite: Install poetry
---------------------------

The entire code is written in python and tested with python 3.11. It uses [poetry](https://python-poetry.org/) to setup a new isolated environment to install all the required libraries. If you haven't installed poetry you could use the make target:

`make download-poetry` 

Or visit the poetry website and do this manually following the instruction at the [poetry website](https://python-poetry.org/docs/#installing-manually) 

Setup the poetry environment
----------------------------

To install all the needed dependencies you can use the make target

`make install`

to install all needed libraries into a new poetry environment.


Training a new model
====================

Before we can start classifying any text into topics we need to train a model. For a jumpstart you can just run

`make run-train`

to start a metaflow run which shouldn't take more than 1 minute to train a new model. You can find the model artifacts in the `./model` subdir.


Details
-------

Our goal is to train a model which can take any text and classify this text into one of the categories: "Menschenrechtsorganisation" or "Umweltschutzorganisation". To do this we need some training data. If you have a closer look at the `./data` subdir you can find a set of wikipedia articles about "Menschenrechtsorganisation" and  "Umweltschutzorganisation" (I literally just crawled some wikipedia categories and dumped each article into a file). 

After we loaded all the training data we need to make this one "usable" for machine-learning models: we need to find a numerical representation for the text. Before we do that we clean up the text a little bit. We are throwing away all non-alphanumeric characters, lowercase each word and stem it (just keep "root" of a word). After doing that we can apply a method call [TF/IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). TF/IDF (Term Frequency/Inverse Document Frequency) is a metric that reflects the importance of a word in a document relative to a collection of documents, considering both the frequency of the word in the document and its rarity across the entire collection.

Once we got that we can transform all the training data into a set of vectors (one document = one vector), assign a category to each vector and train a model on that dataset. In this example we use a so call [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) but there are many other options (Multinomial Naive Bayes, Logistic Regressions or Gradient Boosting Models).

After training a model one would typically want to know how "good" this new model works. To do that a common practise is to split the data before training into a train and a test set. The important part here is the fact that the test-set is NOT used during training. The main reason is to avoid a so called "overfitting" - in simple words the model doesn't learn patterns but just remembers all the examples which makes it perform very poor on unseen data.

To get some insights on the model performance we let the model predict the categories of the test-set and compare the results with the true values. Now we can compute some metrics out of it. I choose: Precision, Recall (also called sensitivity) and Accuracy. Furthermore a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) is generated. After the training you should find a file call `confusion_matrix.png` in the root dir of the repo - looking at this image might enlighten you faster than reading the wikipedia article ;-) .

Last but not least we store the model(s) into the `./model` subdir. After training you should find 2 files: the vectorizer (this one converts text into tf/idf vectors) and a classifier (the SVC model). Those 2 models can we now use to:

Predict campaigns
=================

Again the jump-start should be:

`make run-classify`

This will classify all campaigns from the `./example-campaigns` directory. The campaigns in there are copies from better-place.org. So nothing which the model has seen during training and not even wikipedia articles. Feel free to add more examples (Disclaimers: 1. You should get better results if you have longer texts. 2. The way the model was build it will always give you one of the two categories even if you present "Ulysses" to the model.)


