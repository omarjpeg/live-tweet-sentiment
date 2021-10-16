# live-tweet-sentiment

This is a Streamlit web app that streams two one-word topics using tweepy and PySpark from twitter entered by
the user and runs them through a logisitic regression classification algorithm 
trained on the  [Sentiment140](https://www.kaggle.com/kazanova/sentiment140) dataset,
for more information about how the model was made as well as an exploration/visualization of the dataset,
check this [Jupyter Notebook](https://www.kaggle.com/omarpng/sentiment140-analyzed-and-modelled)

This analysis is then displayed in the streamlit web app page and update whenever new tweets are received

The code is documented to explain the algorithms behind it as much as I could.

## Requirements

Use the package manager pip to install requirements.
```
pip install -r requirements.txt
```
You'll also need PySpark installed on your machine , which also requires JVM installed, for an easier way to run this app install 
[Docker](https://www.docker.com/) and use to pull the docker image that's set up to run the app in a container: 
```
docker pull omardocks/twitter-sentiment-app
```
## Running The App

Navigate using the cmd to the folder where the files are and use:
```
streamlit run app.py
```

If you pulled the docker image prior..
```
docker run omardocks/twitter-sentiment-app
```

## Motivation
This was a internship application project that I built upon and improved as much as I could in my free time as student, testing it multiple times with a lot of use cases and ironing out it's bugs.
I hope you find it useful! Maybe even build upon it and use a better model with it. =)

### Known issues
- Many tweets aren't at all statements about how a person is feeling towards the topic, like ads(especially when querying a brand name) and such and they end up skewing the algorithm since it wasn't trained on such cases
- Model choice might not be the best, a deep learning model would most likely outperform it
- Neutrality threshold is arbitrary (don't know any better way to set it (yet =) )
- Wordcloud can't display non english letters which end up in our tweets even though we query english tweets
- Streamlit app might reset topic names if connection is lost then re-established but the topics entered aren't affected

### Credits
This app uses [emot.py](https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py) by @NeelShah18
Thank you!

