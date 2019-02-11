import numpy as np
import pandas as pd
import nltk
import nltk.tokenize

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


from pathlib import Path

script_location = Path(__file__).absolute().parent
file_location = script_location / 'news-train.csv'

df = pd.read_csv(file_location)
print(nltk.tokenize.word_tokenize(df['HEADER'].values[0]))

# lm = WordNetLemmatizer()

# print(list(map(lm.lemmatize, nltk.tokenize.word_tokenize(df['HEADER'].values[0]))))

sm = PorterStemmer()

text_data = df['HEADER'].values[:10]

docs = list(map(nltk.tokenize.word_tokenize, text_data))
docs_filtered = [list(map(sm.stem, filter(lambda x: (x not in string.punctuation) and (x not in stop_words), doc))) for doc in docs]

print(docs_filtered)

def my_tokenizer(doc):
    tokenized = nltk.tokenize.wordpunct_tokenize(doc)
    res = list(map(sm.stem, filter(lambda x: (x not in string.punctuation) and (x not in stop_words), tokenized)))
    return res

vect = TfidfVectorizer(analyzer='word', tokenizer=my_tokenizer)
print('---Tfidf---')
freqs = vect.fit_transform(text_data)
print(freqs.toarray())
# freqs.toarray()