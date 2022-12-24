import pandas as pd
from torch.utils.data import Dataset
import re
import nltk
from nltk.corpus import stopwords
import sklearn

nltk.download('stopwords')
nltk.download('wordnet')

class TextDataset(Dataset):
    def __init__(self, train=True, seed=421):
        # Reading and preprocessing dataset
        self.data = self.process(pd.read_csv('airline_sentiment_analysis.csv', index_col=0), 'text')

        # Splitting dataset into Train or Test
        if train:
            self.data, _ = sklearn.model_selection.train_test_split(self.data, random_state=seed)
        else:
            _, self.data = sklearn.model_selection.train_test_split(self.data, random_state=seed)
        self.data = self.data.reset_index()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = 0 if self.data['airline_sentiment'][idx] == 'negative' else 1
        text = self.data['text'][idx]
        sample = {'input':text, 'label': label}
        return sample
    
    def process(self, df, col):
        
        def reg(text):
            # Removing links and non alphabets
            text = re.sub(r'https?:\/\/.*[\r]*|#\w+|[^\w\s]|[0-9]*|', '', str(text).lower().strip()) 
            text = re.sub('[ \t]+' , ' ', str(text))
            return text

        def stopWordRemoval(x):
            # Removing Stopwords
            stop = stopwords.words('english')
            x = ' '.join([word for word in str(x).split() if word not in (stop)])
            return x

        def lemmatize_text(text):
            # Removing word stems
            w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
            lemmatizer = nltk.stem.WordNetLemmatizer()
            return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

        df[col] = df[col].apply(reg)
        df[col] = df[col].apply(stopWordRemoval)
        df[col] = df[col].apply(lemmatize_text)
        return df 
