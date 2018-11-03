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
                           names=['label', 'message'])

        counts = self.prepare_data(df)
        x_train, x_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=69)

        self.model = MultinomialNB().fit(x_train, y_train)

    def prepare_data(self, df):
        df['label'] = df.label.map({'positive': 0, 'negative': 1})
        df['message'] = df.message.map(lambda x: x.lower())
        df['message'] = df.message.str.replace('[^\w\s]', '')
        df['message'] = df['message'].apply(nltk.word_tokenize)
        df['message'] = df['message'].apply(lambda x: [self.stemmer.stem(y) for y in x])
        df['message'] = df['message'].apply(lambda x: ' '.join(x))
        counts = self.count_vect.fit_transform(df['message'])
        self.transformer = TfidfTransformer().fit(counts)
        counts = self.transformer.transform(counts)
        return counts

    def predict(self, comment):
        ham_counts = self.count_vect.transform([comment])
        ham_counts = self.transformer.transform(ham_counts)
        result = self.model.predict(ham_counts)
        return result
