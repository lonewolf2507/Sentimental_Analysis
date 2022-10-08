# text preprocessing modules
import string
import numpy as np
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import emoji
import contractions
from deep_translator import GoogleTranslator
from langdetect import detect
from nltk.stem.snowball import SnowballStemmer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import pickle
import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the Airline's reviews",
    version="0.1",
)

# load the sentiment model
model = pickle.load(open('senlr.pkl', 'rb'))
# load tfidf vector
import dill as pickle

with open('tfidf.pickle' ,'rb') as f:
    tf = pickle.load(f)

# cleaning the data
def detect_and_translate(tweet):
    if detect(tweet) == 'en':
        return tweet
    else:
        translated = GoogleTranslator(source='auto', target='en').translate(tweet)
        return translated
def replace_retweet(tweet, default_replace=""):
    tweet = re.sub('RT\s+', default_replace, tweet)
    return tweet

def replace_user(tweet, default_replace="twitteruser"):
    tweet = re.sub('\B@\w+', default_replace, tweet)
    return tweet

def demojize(tweet):
    tweet = emoji.demojize(tweet)
    return tweet

def replace_url(tweet, default_replace=""):
    tweet = re.sub('(http|https):\/\/\S+', default_replace, tweet)
    return tweet

def replace_hashtag(tweet, default_replace=""):
    tweet = re.sub('#+', default_replace, tweet)
    return tweet

def to_lowercase(tweet):
    tweet = tweet.lower()
    return tweet

def word_repetition(tweet):
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
    return tweet

def punct_repetition(tweet, default_replace=""):
    tweet = re.sub(r'[\?\.\!]+(?=[\?\.\!])', default_replace, tweet)
    return tweet

def fix_contractions(tweet):
    tweet = contractions.fix(tweet)
    return tweet

stop_words = set(stopwords.words('english'))
stop_words.discard('not')

def custom_tokenize(tweet, keep_punct = False, keep_alnum = False, keep_stop = False):

    token_list = word_tokenize(tweet)

    if not keep_punct:
        token_list = [token for token in token_list if token not in string.punctuation]
    if not keep_alnum:
        token_list = [token for token in token_list if token.isalpha()]
    if not keep_stop:
        stop_words = set(stopwords.words('english'))
        stop_words.discard('not')
        token_list = [token for token in token_list if not token in stop_words]

    return token_list

def stem_tokens(tokens, stemmer):
    token_list = []
    for token in tokens:
        token_list.append(stemmer.stem(token))
    return token_list

def process_tweet(tweet):
    ## Twitter Features
    tweet = replace_retweet(tweet) # replace retweet
    tweet = replace_user(tweet, "") # replace user tag to null
    tweet = replace_url(tweet) # replace url
    tweet = replace_hashtag(tweet) # replace hashtag
    tweet = detect_and_translate(tweet) # change language to english

  ## Word Features
    tweet = to_lowercase(tweet) # lower case
    tweet = fix_contractions(tweet) # replace contractions
    tweet = punct_repetition(tweet) # replace punctuation repetition
    tweet = word_repetition(tweet) # replace word repetition
    tweet = demojize(tweet) # replace emojis

  ## Tokenization & Stemming
    tokens = custom_tokenize(tweet) # tokenize
    stemmer = SnowballStemmer("english") # define stemmer
    stem = stem_tokens(tokens, stemmer) # stem tokens

    return stem

@app.get("/predict-review")
def predict_sentiment(review: str):
    # A simple function that receive a review content and predict the sentiment of the content.
    # :param review:
    # :return: prediction, probabilities

    # clean the review
    tweet = process_tweet(review)
    # perform prediction
    tweet = tf.transform(tweet)
    # cleaned_review = cleaned_review.reshape(1,-1)
    prediction = model.predict(tweet)
    output = int(prediction[0])
    probas = model.predict_proba(tweet)
    output_probability = "{:.2f}".format(float(np.max(probas[:, output])))

    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}

    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result
