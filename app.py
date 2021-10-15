# importing streamlit as well as NLP libraries for visualization and classification
import streamlit as st
import Listener
import pySpark
from threading import Thread
import pandas as pd, numpy as np, seaborn as sns, pickle, matplotlib.pyplot as plt
import scipy.sparse as sparse
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import re
import time
from wordcloud import WordCloud
from emo_unicode import *
from nltk.corpus import wordnet as wn

# set a button clicked flag for our streamlit app
buttonclicked = False
sns.set()

# a dict to process/clean tweets
processing_dict = {
    # replace links with URL
    r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)": 'URL',
    # delete any HTML special Entities
    r"&\w{2,5};": ' ',
    # replace mentions with user
    r"@[^\s]+": 'USER',
    # fix typos
    r"(.)\1\1+": r"\1\1"
    # for deeper insight into why this approach was taken check https://github.com/omarjpeg/sentiment140
}


def update():
    # setting up the layout of the streamlit app, and providing empty "areas" for the visualization figures
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        area1 = st.empty()
        underarea1 = st.empty()
    with col2:
        areamid = st.empty()
    with col3:
        area2 = st.empty()
        underarea2 = st.empty()
    st.write(
        'The model is trained on the Sentiment140 dataset to classify incoming tweets and has a testing score of 80%!')
    st.write('by Omar Ayman  '
             ' https://www.linkedin.com/in/omar-png/')
    st.write('*Most common words may include some two to three character, this is a result of lemmetization. The more popular the topic, the more the tweets are and the longer it will take to process, check the command line for current progress')
    while True:
        # update visualizations every 10 seconds for newer tweets received
        time.sleep(10)
        tweets = read_clean_analyze_tweet()
        if not tweets.empty:
            update_figures(tweets, area1, underarea1, area2, underarea2, areamid)


def analyze_tweet(txt, time):
    # to learn how this algorithm was made check https://github.com/omarjpeg/sentiment140
    # basically preparing data to be fed to a logistic reg algorithm that relies on tfidf vectorized text and time data(day of week and time of day)
    week = np.zeros(31)
    time = pd.to_datetime(time)
    week[time.weekday()] = 1
    week[time.hour] = 1
    mx = loaded_vect.transform([txt])
    pos, neg = loaded_model.predict_proba(sparse.hstack((mx, sparse.csr_matrix(week))))[0]
    read_clean_analyze_tweet.tweetcounter += 1
    #progress shown in cmd
    print('analyzed {}/{} tweets'.format(read_clean_analyze_tweet.tweetcounter,read_clean_analyze_tweet.totaltweets))
    if read_clean_analyze_tweet.tweetcounter == read_clean_analyze_tweet.totaltweets:
        print("Updating figures..")
    # arbitrary neutrality threshold, don't know any better (for now =) )
    if 0.58 >= pos >= 0.42:
        return 1
    elif pos > 0.52:
        return 2
    else:
        return 0


# set max width for streamlit app using html since it can get too wide on certain screens
def reduce_max_width():
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


def remove_stopwords(text):
    stopwordsfree = " ".join([x for x in text.split() if x not in stop])
    return stopwordsfree


def lemmetize_sentence(text):
    word_list = nltk.word_tokenize(text)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output


def clean_text(txt):
    # anything but text
    txt = re.sub(r'[^a-zA-Z0-9]', ' ', txt)
    return txt


def process_text(txt):
    # removing any emoticons or unicode emojis using the
    # help of https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py by Neel & contributors, Thanks!

    for feeling, code in EMOJI_UNICODE.items():
        txt = txt.replace(code, clean_text(feeling))
    for toreplace, replacement in processing_dict.items():
        txt = re.sub(toreplace, replacement, txt)
    for face, feeling in EMOTICONS_EMO.items():
        txt = txt.replace(face, feeling)
    txt = txt.lower()
    return txt


def read_clean_analyze_tweet():
    # read the updated tweets saved by pyspark process and analyze/classify them
    tweets = pd.read_csv('current_tweets.txt', encoding='utf8', sep='__TIME_END__', names=['time', 'tweet'])
    # do analysis/classification only if there are tweets about he topic
    tweets = tweets[tweets.tweet != None]
    #to show in console as a "loading screen"
    read_clean_analyze_tweet.tweetcounter = 0
    read_clean_analyze_tweet.totaltweets =tweets.shape[0]
    if not tweets.empty:
        tweets['display_words'] = tweets.tweet.apply(lambda x: process_text(x))
        tweets.tweet = tweets.tweet.apply(lambda x: clean_text(remove_stopwords(lemmetize_sentence(x))))
        # what if tweet was only symbols? drop it
        tweets = tweets[tweets['tweet'] != ' ']
        tweets['score'] = tweets.apply(lambda x: analyze_tweet(x.tweet, x.time), axis=1)
    return tweets


def update_figures(tweets, area1, underarea1, area2, underarea2, areamid):
    # know which tweets belong to which topic, tweets mentioning both topics are shared

    topic1tweets = tweets[tweets.display_words.str.contains(topic1.lower())]
    topic2tweets = tweets[tweets.display_words.str.contains(topic2.lower())]

    labels = ['Negative', 'Neutral', 'Positive']
    explode = (0, 0.1, 0.085)

    # update pie chart for every pie chart figure showing how percentages of each sentiment
    for area, topic, data in [(area1, topic1, topic1tweets), (area2, topic2, topic2tweets)]:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.axis('equal')
        ax.text(1 if topic == topic1 else 0, 0, f'Based on {data.shape[0]:,} Tweet(s)', transform=ax.transAxes)
        srs = pd.Series(data=[0, 0, 0], index=[0, 1, 2]).add(data.score.value_counts(), fill_value=0)
        data = pd.Series(data=srs, index=srs.index)
        ax.pie(data, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title(topic.capitalize(), fontsize=24, horizontalalignment='center')
        area.pyplot(fig)

    # show tweet relevancy by a bar plot of tweet count
    fig3, ax3 = plt.subplots(figsize=(2, 6), constrained_layout=True)
    data = [topic1tweets.shape[0], topic2tweets.shape[0]]
    bars = ax3.bar([topic1.capitalize(), topic2.capitalize()], data, width=0.34)
    ax3.set_facecolor('white')
    fig3.suptitle('Relevancy', fontsize=12)
    ax3.set_yticks([])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + .01, f'{yval:,}\nTweet(s)', fontsize=10, ha='center')
    areamid.pyplot(fig3)

    # update word cloud for every word cloud figure showing most common words except list of words that are passed
    for area, stop_words, data in [(underarea1, dontshow1, topic1tweets), (underarea2, dontshow2, topic2tweets)]:
        wordfig, w1ax1 = plt.subplots(constrained_layout=True, figsize=(2, 1))
        w1ax1.grid(False)
        w1ax1.set_xticks([])
        w1ax1.set_yticks([])
        if not data.empty:
            wc = WordCloud(background_color='white', max_words=500, width=1000, height=500, collocations=False,
                           stopwords=set(stop_words)).generate(
                " ".join(data.display_words))
            w1ax1.imshow(wc, interpolation='bilinear')
            w1ax1.set_title("Most Frequent Words by Size*", fontsize=4)
        area.pyplot(wordfig)


# Load model and vectorizer and init streamlit page
model_name = 'logistic_model.pk'
vectorizer_name = 'tfidf_vectorizer.pk'
loaded_model = pickle.load(open(model_name, 'rb'))
loaded_vect = pickle.load(open(vectorizer_name, 'rb'))
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'>Sentimental Analysis of Tweets by Topic</h1>",
            unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>This app is designed to take 2 topics "
            "(as words) and compare them in terms of positive to negative tweets!</h3>", unsafe_allow_html=True)

reduce_max_width()
col1, col2 = st.columns([1, 1])

# fill user text boxes with default generic topics
with col1:
    topic1 = st.text_input('Topic #1:', value="Spiderman")
with col2:
    topic2 = st.text_input('Topic #2:', value="Batman")

# activated when user presses go button
if st.button('Go!'):
    # if not pressed prior set previously set flag to True making it unusable
    if not buttonclicked:
        buttonclicked = True
        wn.ensure_loaded()  # prevents errors working with wordnet
        stop = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        # write current tweets file if not present already, then close it
        open('current_tweets.txt', 'w').close()
        # start both tweet listener py file and pyspark tweet receiver py file to launch on differnt threads as a background process seperate from the UI
        thread2 = Thread(target=Listener.start_listening_and_send_tweets, args=[[topic1, topic2]])
        thread = Thread(target=pySpark.receive_stream, args=[])
        thread2.start()
        thread.start()
        # Inform the user that it started
        st.write(
            'Below you will find two LIVE updating (each 10 second interval, give it some time to  load! (= ) '
            'pie charts comparing the two terms you just entered and how the sentimental analysis algorithm is '
            'classifying them!')

        # Since the app is designed to work with one word topics, this deals with names like Mcdonald's showing up as mcdonald in the wordcloud due to
        # an issue where words are forcibly processed to remove apostrophes without an option to disable it, or atleast i couldn't find it
        topic1noapostrophe = topic1[:-2] if topic1[-1:-3:-1] == "s'" else topic1
        topic2noapostrophe = topic2[:-2] if topic2[-1:-3:-1] == "s'" else topic2

        # stop words not to show, including user mentions, urls and the topic itself but not the compared topic
        # this is ommitted to prevent USER and URL clutter on the wordcloud but can be interesting to observe
        dontshow1 = topic1.split() + ['url', 'user'] + list(stopwords.words('english')) + [topic1noapostrophe]
        dontshow2 = topic2.split() + ['url', 'user'] + list(stopwords.words('english')) + [topic2noapostrophe]

        update()
