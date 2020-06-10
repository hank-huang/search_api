import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')


def preprocess(sentence):
    stemmer = PorterStemmer()

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)

    sentence = [w for w in word_tokens if w not in stop_words]

    output = []

    for w in sentence:
        output.append(stemmer.stem(w))

    return output
