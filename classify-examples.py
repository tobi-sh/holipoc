import pickle
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('german')

def classify_examples():
    """classify all files at example-campaigns
    """
    print("Loading examples")
    examples = load_examples()
    classifier = load_pickle('classifier.pkl')
    vectorizer = load_pickle('vectorizer.pkl')
    
    for (name, doc) in examples.items():
        features = vectorizer.transform([tokenize_and_stem(doc)])
        print(f'{name} => {classifier.predict(features)}')


def load_pickle(filename):
    print(f"Loading {filename}")
    with open(filename, 'rb') as file:
        return pickle.load(file)

def load_examples():
    examples = {}
    for root, dirs, files in os.walk('example-campaigns'):
            for fil in files:
                with open(os.path.join(root, fil), 'r') as f:
                    examples[fil] = f.read()
    return examples

def tokenize_and_stem(doc):
        tokens = word_tokenize(doc, language='german')
        return ' '.join([stemmer.stem(token) for token in tokens if token.isalnum()])
    
if __name__ == '__main__':
    classify_examples()