from __future__ import print_function
from pyspark.streaming import StreamingContext
import socket
from pyspark import SparkContext
import pandas as pd

host_name = socket.gethostbyname(socket.gethostname())


def function(c):
    pd.set_option("display.max_colwidth", -1)
    old_tweets = pd.read_csv('current_tweets.txt',encoding = 'utf8',delimiter = '\n', names=['tweet'],dtype='str')
    new_tweets = pd.DataFrame({'tweet': c.collect()},dtype='str').drop_duplicates()
    to_add= new_tweets.merge(old_tweets,on='tweet',how= 'left',indicator = True
                             ).query('_merge == "left_only"').drop('_merge', 1)
    with open('current_tweets.txt', 'a+', encoding='utf8', errors="ignore") as file:
        for item in to_add.tweet:
            file.write("%s\n" % item)
    return c


def receive_stream():
    sc = SparkContext(appName="StreamTwitter")
    sc.setLogLevel("Error")
    ssc = StreamingContext(sc, 10)

    socket_stream = ssc.socketTextStream(host_name, 5555)

    lines = socket_stream.window(10)

    lines.foreachRDD(lambda x: function(x))

    ssc.start()

    ssc.awaitTermination()


