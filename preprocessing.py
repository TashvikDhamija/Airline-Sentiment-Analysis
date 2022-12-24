import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')

def process(text):

    def reg(text):
        # removing links and non alphabets
        text = re.sub(r'https?:\/\/.*[\r]*|#\w+|[^\w\s]|[0-9]*|', '', str(text).lower().strip())
        text = re.sub('[ \t]+' , ' ', str(text))
        return text

    def stopWordRemoval(x):
        # removing stopwords
        stop = stopwords.words('english')
        x = ' '.join([word for word in str(x).split() if word not in (stop)])
        return x

    def lemmatize_text(text):
        # removing word stems
        w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

    text = lemmatize_text(stopWordRemoval(reg(text)))
    return text