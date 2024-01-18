import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pickle

import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from metaflow import FlowSpec, step
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ArticleClassifier(FlowSpec):
    
    
    
    @step
    def start(self):
        print('downloading prerequists')
        nltk.download('punkt') 
        self.stemmer = SnowballStemmer('german')
        self.next(self.load_data)
    
    @step    
    def load_data(self):
        self.article_text = []
        self.categories = []
        for root, dirs, files in os.walk('data'):
            for fil in files:
                with open(os.path.join(root, fil), 'r') as f:
                    self.categories.append(root[5:])
                    self.article_text.append(f.read())
        print(f'Loaded {len(self.article_text)} documents')
        self.next(self.preprocess_text)
    
    @step
    def preprocess_text(self):
        self.tokenized_articles = [self._tokenize_and_stem(article) for article in self.article_text]
        self.next(self.fit_vectorizer)
        
    @step 
    def fit_vectorizer(self):
        self.vectorizer = TfidfVectorizer(max_df=0.7, stop_words=stopwords.words('german'))
        self.tokenized_articles = self.vectorizer.fit_transform(self.tokenized_articles)
        self.next(self.train_classifier)
        
    @step 
    def train_classifier(self):
        self.classifier = SVC(kernel='linear')
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.tokenized_articles, self.categories, test_size=0.2, random_state=240480)
        self.classifier.fit(X_train, y_train)
        self.next(self.persist)
    
    @step
    def persist(self):
        # safe the classifier:
        model_filename = 'classifier.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(self.classifier, file)
            
        # safe the vecorizer:
        model_filename = 'vecorizer.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(self.vectorizer, file)
            
        self.next(self.print_metrics)
        
    @step
    def print_metrics(self):
        y_pred = self.classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        for label in set(self.categories):
            precision = precision_score(self.y_test, y_pred, pos_label=label)
            recall = recall_score(self.y_test, y_pred, pos_label=label)
            print(f"Precision for {label}: {precision:.2f}")
            print(f"Recall for {label}: {recall:.2f}")
        
        print(f"Accuracy: {accuracy:.2f}")

        print(f'Confussion Matrix (Test-Set):\n{cm}')
        
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classifier.classes_)
        
        # Plot the confusion matrix
        cm_display.plot(cmap='Blues', values_format='d')

        # Save the figure to a PNG file
        plt.savefig('confusion_matrix.png', format='png', bbox_inches='tight')



        self.next(self.show_important_features)

    @step
    def show_important_features(self):
        # Get the feature names (tokens)
        feature_names = self.vectorizer.get_feature_names_out()

        # Get the coefficients (weights) of the SVC model
        coefficients = self.classifier.coef_.toarray()[0]

        # Create a dictionary of feature names and their corresponding coefficients
        feature_coefficients = dict(zip(feature_names, coefficients))

        # Sort the features by their coefficients (weights)
        sorted_features = sorted(feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True)

        # Print the top N features with the highest impact
        print(f"Top 20 features with the highest impact:")
        for feature, coefficient in sorted_features[:20]:
            print(f"{feature}: {coefficient}")
        self.next(self.end)

    @step
    def end(self):
        print('I am done')
        
    def _tokenize_and_stem(self, doc):
        tokens = word_tokenize(doc, language='german')
        return ' '.join([self.stemmer.stem(token) for token in tokens if token.isalnum()])

if __name__ == '__main__':
    ArticleClassifier()
