from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from multiprocessing import Pool
from sklearn.linear_model import SGDClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from bs4 import BeautifulSoup
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import string
import nltk
import os

customStopWords = ["movie", "film", "films"]


folder = 'aclImdb'
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()
for f in ('test', 'train'):    
    for l in ('pos', 'neg'):
        path = os.path.join(folder, f, l)
        for file in os.listdir (path) :
            with open(os.path.join(path, file),'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],ignore_index=True)
df.columns = ['review', 'sentiment']

'''
#function to split text into word
tokens = None
if os.path.exists('tokens.pickle'):
    with open('tokens.pickle', 'rb') as handle:
        tokens = pickle.load(handle)
else:
    reviews = df.review.str.cat(sep=' ')
    soup = BeautifulSoup(reviews)
    reviews = soup.get_text()
    reviews = reviews.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(reviews)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if not w in customStopWords]
    # lemmatize words
    lmtzr = WordNetLemmatizer()
    words = [lmtzr.lemmatize(w) for w in words]
    
    tokens = words
    with open('tokens.pickle', 'wb') as handle:
        pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)

vocabulary = set(tokens)
print(len(vocabulary))

frequency_dist = nltk.FreqDist(tokens)
print(sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50])


wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
'''
X_train = df.loc[:24999, 'review'].values
y_train = df.loc[:24999, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

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
    words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(reviewText)]
    reviewText = ' '.join([lemmatizer.lemmatize(w) for w in words])
    reviewText = reviewText.lower()
    reviewText = reviewText.translate(str.maketrans('', '', string.punctuation))
    return reviewText

if __name__ == '__main__':
    with Pool(12) as p:
        X_train = p.map(prepText, X_train)

if __name__ == '__main__':
    with Pool(12) as p:
        X_test = p.map(prepText, X_test)

print(X_train[10:])

print("Vectorizing")

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)

clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=200, tol=None).fit(train_vectors, y_train)

predicted = clf.predict(test_vectors)
#print(accuracy_score(y_test,predicted))
print(metrics.classification_report(y_test, predicted))
while True:
    userReview = prepList([input("Enter review here: ")])
    userVector = vectorizer.transform(userReview)
    predicted = clf.predict(userVector)
    if predicted[0] == 1:
        print("This was a positive review!")
    if predicted[0] == 0:
        print("This was a negative review!")
    shouldContinue = input("Submit another review? (Y/N)")
    if shouldContinue == "N":
        break
