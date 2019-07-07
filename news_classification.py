from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SpatialDropout1D
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Bidirectional
from multiprocessing import Pool
from keras.callbacks import EarlyStopping
from nltk.corpus import stopwords
from sklearn import metrics
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import keras
import nltk
import pandas as pd
import pickle
import string
import os

if __name__ == '__main__':
    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 500
    # This is fixed.
    EMBEDDING_DIM = 100

    folder = '20news-18828'

    traindf = pd.DataFrame()
    #testdf = pd.DataFrame()
    def get_immediate_subdirectories(a_dir):
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    categories = get_immediate_subdirectories(folder)
    for category in categories:
        path = os.path.join(folder, category)
        count = 0
        for file in os.listdir (path) :
            with open(os.path.join(path, file),'r') as infile:
                txt = infile.read()
            count += 1
            if count < 1100:
                traindf = traindf.append([[txt, category]], ignore_index=True)
            else:
                testdf = testdf.append([[txt, category]], ignore_index=True)

    traindf.columns = ['articles', 'category']
#testdf.columns = ['articles', 'category']

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()

def prepText(reviewText):
    soup = BeautifulSoup(reviewText)
    reviewText = soup.get_text()
    words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_tokenize(reviewText)]
    stop_words = set(stopwords.words('english')) 
    words = [w for w in words if not w in stop_words] 
    reviewText = ' '.join([lemmatizer.lemmatize(w) for w in words])
    reviewText = reviewText.lower()
    reviewText = reviewText.translate(str.maketrans('', '', string.punctuation))
    return reviewText

#print(X_train)
if __name__ == '__main__':
    method = input("Enter method number: ")
    if method == "1":
        X_train = traindf.loc[:, 'articles'].values
        y_train = traindf.loc[:, 'category'].values
        X_test = testdf.loc[:, 'articles'].values
        y_test = testdf.loc[:, 'category'].values
        vectorizer = TfidfVectorizer()
        train_vectors = vectorizer.fit_transform(X_train)
        test_vectors = vectorizer.transform(X_test)
        print(train_vectors.shape, test_vectors.shape)

        clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=200, tol=None).fit(train_vectors, y_train)
        predicted = clf.predict(test_vectors)
        print(metrics.classification_report(y_test, predicted))
    if method == "2":
        if os.path.exists('articles.pickle'):
            filehandler = open("articles.pickle", "rb")
            traindf['articles'] = pickle.load(filehandler)
        else:
            with Pool(12) as p:
                traindf['articles'] = p.map(prepText, traindf['articles'])
            filehandler = open("articles.pickle", 'wb') 
            pickle.dump(traindf['articles'], filehandler)
        print("Using LSTM")
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(traindf['articles'].values)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        X = tokenizer.texts_to_sequences(traindf['articles'].values)
        X = keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        Y = pd.get_dummies(traindf['category']).values
        print('Shape of data tensor:', X.shape)
        print('Shape of label tensor:', Y.shape)
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
        print(X_train.shape,Y_train.shape)
        print(X_test.shape,Y_test.shape)
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(LSTM(150, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(20, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        epochs = 5
        batch_size = 64

        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        accr = model.evaluate(X_test,Y_test)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        plt.title('Accuracy')
        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='test')
        plt.legend()
        plt.show() 