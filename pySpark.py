from pyspark.streaming import StreamingContext
import socket
from pyspark import SparkContext
import pandas as pd
#get local ip
host_name = socket.gethostbyname(socket.gethostname())


def process_save_tweets(c):
    # twitter api can send repeated tweets, and an app that uses it must take that into consideration
    # here we read old tweets and make sure no duplicates exist in our dataset
    old_tweets = pd.read_csv('current_tweets.txt', encoding='utf8', delimiter='\n', names=['tweet'], dtype='str')
    new_tweets = pd.DataFrame({'tweet': c.collect()}, dtype='str').drop_duplicates()

    # we use a LEFT-Excluding JOIN to get new tweets only
    to_add = new_tweets.merge(old_tweets, on='tweet', how='left', indicator=True
                              ).query('_merge == "left_only"').drop('_merge', 1)
    #append the new tweets to our text file, leaving old ones alone
    with open('current_tweets.txt', 'a+', encoding='utf8', errors="ignore") as file:
        for item in to_add.tweet:
            file.write("%s\n" % item)
    return c


def receive_stream():
    # spark context init
    sc = SparkContext(appName="StreamTwitter")
    sc.setLogLevel("Error")

    # pass our spark context to the streaming context and set batch duration to every 10 seconds
    # every 10 seconds, data received for every 10 seconds is packaged as a new batch to be processed
    ssc = StreamingContext(sc, 10)

    #init a socket text stream to listen on port 5555
    socket_stream = ssc.socketTextStream(host_name, 5555)

    # set the incoming stream to return a processed Dstream of rdds every 10 seconds(meaning a new batch will be read) which we will process immediately
    lines = socket_stream.window(10)

    # this function occurs every 10 seconds, whenever a new dstream is returned filled with a new data batch
    lines.foreachRDD(lambda x: process_save_tweets(x))

    # start listening for accompanying TweetsListener class to send tweets with the send_tweets function on port 5555
    ssc.start()

    #keep listening till process is killed
    ssc.awaitTermination()

