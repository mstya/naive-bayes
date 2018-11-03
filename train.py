import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


class NaiveBayes:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.count_vect = CountVectorizer()
        self.transformer = {}
        self.model = {}

    def train(self, file):
        df = pd.read_table(file,
                           sep='\t',
                           header=None,
                           names=['label', 'comment'])

        counts = self.prepare_data(df)
        x_train, x_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=69)

        self.model = MultinomialNB().fit(x_train, y_train)

    def prepare_data(self, df):
        df['label'] = df.label.map({'positive': 0, 'negative': 1})
        df['comment'] = df.comment.map(lambda x: x.lower())
        df['comment'] = df.comment.str.replace('[^\w\s]', '')
        df['comment'] = df['comment'].apply(nltk.word_tokenize)
        df['comment'] = df['comment'].apply(lambda x: [self.stemmer.stem(y) for y in x])
        df['comment'] = df['comment'].apply(lambda x: ' '.join(x))
        counts = self.count_vect.fit_transform(df['comment'])
        self.transformer = TfidfTransformer().fit(counts)
        counts = self.transformer.transform(counts)
        return counts

    def predict(self, comment):
        counts = self.count_vect.transform([comment])
        counts = self.transformer.transform(counts)
        result = self.model.predict(counts)
        return result
