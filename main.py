import gensim as gensim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import warnings
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from random import uniform
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
import datetime
import calendar

# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
# from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import to_categorical
# import re

warnings.filterwarnings('ignore')

def main():
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1')
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

    # pd.set_option('display.max_columns', None)

    work_df = df[799500:800500]     #reducing the sample size for speed (specific choice so we have 0 and 4 as target)

    data_preprocessing(work_df)       #prepares the dataset with some processing
    # visualization(work_df)      #visualizes the words in dataset
    # hashtag_frequency(work_df)        #graphs the hashtag frequency

    vectors = None

    q = input('Press 1 for CountVectorizer or 2 for Word2Vec: ')
    if q == 1:
        vectors = vectorizer(work_df)       #converts text to vectors
    elif q == 2:
        vectors = wordtovec(work_df['clean'])       #use of word2vec libray

    work_df = date_prep(work_df)        #splits date to year, month, day, hour, minute, second

    work_df = work_df.drop(['ids', 'date', 'flag', 'user', 'text'], axis='columns')     #drops the unwanted columns

    a = input('Do you want to include dates & time to the prediction? y/n: ')
    if a == 'y':
        try:
            X = work_df.drop(['target', 'clean'], axis='columns')
            X = pd.concat([pd.DataFrame(vectors).reset_index(drop=True), X.reset_index(drop=True)], axis='columns', ignore_index=True)        #dataframe with vectors + dates
            model_training(X, work_df['target'])        #sklearn model for training and prediction
        except:
            pass
    elif a== 'n':
        try:
            model_training(vectors, work_df['target'])        #sklearn model for training and prediction
        except:
            pass


    # lstm(vectors, work_df['target'])        #keras lstm

# def lstm(bow, target):
#     embed_dim = 128
#     lstm_out = 196
#     max_features = 2000
#
#     model = Sequential()
#     model.add(Embedding(max_features, embed_dim,input_length = bow.shape[1]))
#     model.add(SpatialDropout1D(0.4))
#     model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
#     model.add(Dense(2, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # print(model.summary())
#
#     X_train, X_test, Y_train, Y_test = train_test_split(bow, target, test_size=0.33, random_state=42)
#     # print(X_train.shape, Y_train.shape)
#     # print(X_test.shape, Y_test.shape)
#
#     batch_size = 32
#     model.fit(X_train, Y_train, epochs=7, batch_size=batch_size, verbose=2)
#
#     validation_size = 1500
#
#     X_validate = X_test[-validation_size:]
#     Y_validate = Y_test[-validation_size:]
#     X_test = X_test[:-validation_size]
#     Y_test = Y_test[:-validation_size]
#     score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
#     print("score: %.2f" % (score))
#     print("acc: %.2f" % (acc))

def wordtovec(tweets):
    # import logging
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # import gensim.downloader as api
    # corpus = api.load('text8')
    # import inspect
    # print(inspect.getsource(corpus.__class__))
    # print(inspect.getfile(corpus.__class__))
    # model = Word2Vec(corpus)
    # model.save('./readyvocab.model')

    model = Word2Vec.load('readyvocab.model')

    processed_sentences = []
    for sentence in tweets:
        processed_sentences.append(gensim.utils.simple_preprocess(sentence))

    vectors = {}
    i = 0
    for v in processed_sentences:
        vectors[str(i)] = []
        for k in v:
            try:
                vectors[str(i)].append(model.wv[k].mean())
            except:
                vectors[str(i)].append(np.nan)
        i += 1

    df_input = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in vectors.items()]))
    df_input.fillna(value=0.0, inplace=True)  # oti einai nan paei 0

    df_input = df_input.transpose()

    return df_input

def date_prep(work_df):
    work_df['date_year'] = work_df['date'].dt.year
    work_df['date_month'] = work_df['date'].dt.month
    work_df['date_day'] = work_df['date'].dt.day
    work_df['date_hour'] = work_df['date'].dt.hour
    work_df['date_min'] = work_df['date'].dt.minute
    work_df['date_sec'] = work_df['date'].dt.second

    return work_df

def vectorizer(tweets):
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    bow = bow_vectorizer.fit_transform(tweets['clean'])      #creates a  sparse matrix (mostly zeros) with words

    vec = pd.DataFrame.sparse.from_spmatrix(bow)
    # print(vec)

    return vec
    # return bow

def model_training(bow, target):
    X_train, X_test, y_train, y_test = train_test_split(bow, target, test_size=0.3, random_state=42)     #splits df in train, test, random_state=42 so it splits with the same way each time

    #Logistic Regression
    log = LogisticRegression()
                                        #hyperparameter tuning
    distributions = dict(       #dictionary with parameters of logistic regression
        solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        penalty=['none', 'l1', 'l2', 'elasticnet']
        )

    random_search_cv = RandomizedSearchCV(      #random search between above parameters to find the best
        estimator=log,
        param_distributions=distributions,
        cv=5,
        n_iter=50       #how many times it checks them
        )

    random_search_cv.fit(X_train, y_train)
    print(f'\nBest parameters for Logistic Regression: {random_search_cv.best_params_}')

    log = random_search_cv.best_estimator_

    pred = log.predict(X_test)
    print('\nf1_score: ', f1_score(y_test, pred, pos_label=0))
    print('log_score: ', log.score(X_test, y_test))
    print('accuracy_score: ', accuracy_score(y_test, pred))
    # print('Probability to be: \n   Positive / Negative: \n', log.predict_proba(X_test))

    print('\nScores after probability rework: ')
    pred_proba = log.predict_proba(X_test)      #we try to increase score
    pred2 = pred_proba[:, 0] >= 0.35      #>35% positive counts as positive instead of >50%
    pred2 = pred.astype(np.int)
    print('f1_score: ', f1_score(y_test, pred2, average='micro', pos_label=0))
    print('log_score: ', log.score(X_test, y_test))
    print('accuracy_score: ', accuracy_score(y_test, pred2))

    #Decision Tree Classifier
    print('\nDecision Tree Classifier:')
    scores = cross_val_score(DecisionTreeClassifier(), X_train, y_train, cv=5)  #5 different parameters test
    # print(scores)
    print('\nMean DTC score: ', scores.mean(), sep='')

    bag_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),       #100 models with different samples and takes the majority vote for score
                                  n_estimators=100,
                                  max_samples=0.8,
                                  oob_score=True,  # oob=out of bag
                                  random_state=0)

    bag_model.fit(X_train, y_train)
    print('Bagging score: ', bag_model.score(X_test, y_test), sep='')

    print('Mean Bagging score: ', cross_val_score(bag_model, X_train, y_train, cv=5).mean(), sep='')

    from sklearn.ensemble import RandomForestClassifier
    print('Mean Random Forest score: ', cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5).mean(), sep='')

def hashtag_frequency(tweets_df):
    ht_positive = hashtag_extract(tweets_df['clean'][tweets_df['target']==4])       #thetika hashtags
    ht_negative = hashtag_extract(tweets_df['clean'][tweets_df['target']==0])       #arnitika hashtags

    ht_positive = sum(ht_positive, [])
    ht_negative = sum(ht_negative, [])
    q = input('Would you like to see the hashtag frequency of top hashtags? y/n: ')
    if q == 'y':
        a = input('Press ~p~ for positive or ~n~ for negative: ')
        if a == 'p':
            freq = nltk.FreqDist(ht_positive)
            d = pd.DataFrame({'Hashtag': list(freq.keys()),
                              'Count': list(freq.values())})
            h = input('How many hashtags to show? Type a number: ')
            d = d.nlargest(columns='Count', n=int(h))
            plt.figure(figsize=(15,9))
            sns.barplot(data=d, x='Hashtag', y='Count')
            plt.show()
        elif a == 'n':
            freq = nltk.FreqDist(ht_negative)
            d = pd.DataFrame({'Hashtag': list(freq.keys()),
                              'Count': list(freq.values())})
            h = input('How many hashtags to show? Type a number: ')
            d = d.nlargest(columns='Count', n=int(h))
            plt.figure(figsize=(15, 9))
            sns.barplot(data=d, x='Hashtag', y='Count')
            plt.show()
    else:
        pass

def hashtag_extract(tweets):
    hashtags = []
    for tweet in tweets:
        ht = re.findall(r'#(\w+)', tweet)   #finds hashtags
        hashtags.append(ht)     #appends them

    return hashtags

def visualization(tweets_df):
    q = input('Would you like to visualize your data? y/n: ')
    if q == 'y':
        a = input('Press ~p~ for positive or ~n~ for negative: ')
        if a == 'p':
            all_words = ' '.join([sentence for sentence in tweets_df['clean'][tweets_df['target']==4]])     #puts all words together
            wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)        #loads them to wordcloud

            plt.figure(figsize=(15, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()
        elif a == 'n':
            all_words = ' '.join([sentence for sentence in tweets_df['clean'][tweets_df['target']==0]])
            wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

            plt.figure(figsize=(15, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()
    else:
        pass

def data_preprocessing(tweets_df):

    date_trash = ['PDT', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    months = {'Jan': '1', 'Feb': '2', 'Apr': '3', 'Mar': '4', 'May': '5', 'Jun': '6', 'Jul': '7', 'Aug': '8', 'Sep': '9', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

    for x in date_trash:
        tweets_df['date'] = tweets_df['date'].str.replace(x, '')      #deletes name of days and PDT

    for k, v in months.items():
        tweets_df.date = tweets_df.date.str.replace(k, v)       #converts months to numbers with the help of dictionary created above

    tweets_df['date'] = tweets_df['date'].apply(lambda x: datetime.datetime.strptime(x, ' %m %d %H:%M:%S  %Y') if type(x)==str else np.NaN)     #converts to datetime
    # print(tweets_df['date'])

    tweets_df['clean'] = np.vectorize(remove_pattern)(tweets_df['text'], "@[\w]*")  #deletes @user from tweets
    tweets_df['clean'] = tweets_df['clean'].str.replace('[^a-zA-Z#]', ' ')  # deletes special characters like :
    tweets_df['clean'] = tweets_df['clean'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 3]))  # deletes words with < 3 letters
    # print(tweets_df[:10])

    token_tweets = tweets_df['clean'].apply(lambda x: x.split())    #converts words to tokens for easier processing
    # print(token_tweets[:10])

    stemmer = PorterStemmer()
    token_tweets = token_tweets.apply(lambda sentence: [stemmer.stem(word) for word in sentence])  # similar words are converted into the shortest similar word
    # print('\ntoken tweets', token_tweets[:10])

    for i in range(len(token_tweets)):
        token_tweets.iloc[i] = ' '.join(token_tweets.iloc[i])

    tweets_df['clean'] = token_tweets
    # print(tweets_df[:10])

    return tweets_df

def remove_pattern(tweets, pattern):    #deletes what we insert
    r = re.findall(pattern, tweets)
    for word in r:
        tweets = re.sub(word, '', tweets)
    return tweets


if __name__ == '__main__':
    main()
