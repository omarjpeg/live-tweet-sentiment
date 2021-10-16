from tweepy import Stream
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import json
import socket

#twitter api keys, you have to sign up for a twitter developer account to get these
consumer_key = 'AYoR5fpYK5IZuvHJtCBNd7tjx'
consumer_secret = 'XXqvJtQ8HOZyls19SbgWNYQG6MguLDkA8qtMFFrMw1uTA2OWhD'
access_token = '1438526381109833735-mBSov4dGzXLMaaHRtKGJ4xw6O77sGH'
access_secret = 'zD4d9XGCSxvy1uKA5Nv4cWlRcFse9ZKwuzvLrV7oSS9r4'
#get local ip
host_name = socket.gethostbyname(socket.gethostname())


class TweetsListener(StreamListener):
    # tweet object listens for the tweets
    def __init__(self, csocket):
        super().__init__()
        self.client_socket = csocket

    def on_data(self, data):
        try:
            msg = json.loads(data)
            # add at the end of each tweet time "__TIME_END__" to load later for our algorithm
            # if tweet is longer than 140 characters
            if "extended_tweet" in msg:
                self.client_socket \
                    .send((str(msg['created_at']) + '__TIME_END__' + str(msg['extended_tweet']['full_text']).replace(
                    "\n", " ") + '\n'
                           ).encode('utf-8'))
            elif "retweeted_status" in msg:
                #handle retweets by getting the text said in the retweet text that mentions the topic, ignoring the retweeted tweet
                if "extended_tweet" in msg["retweeted_status"]:
                    self.client_socket \
                    .send((str(msg['retweeted_status']['created_at']) + '__TIME_END__' + str(msg['retweeted_status']['extended_tweet']['full_text']).replace(
                    "\n", " ") + '\n'
                           ).encode('utf-8'))
                else:
                    #same as above, but with no extended_tweet attribute so we mention text instead, returning retweet text only
                    self.client_socket \
                        .send((str(msg['retweeted_status']['created_at']) + '__TIME_END__' + str(
                        msg['retweeted_status']['text']).replace(
                        "\n", " ") + '\n'
                               ).encode('utf-8'))
            else:
                # add at the end of each tweet "t_end"
                self.client_socket \
                    .send((str(msg['created_at']) + '__TIME_END__' + str(msg['text']).replace("\n", " ") + '\n') \
                          .encode('utf-8'))
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True


def sendData(c_socket, keyword):
    print('started sending data from Twitter to socket')
    # authentication based on the credentials
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    # start sending data from the Streaming API to the socket passed, filtered by passed list of keyword(s)
    twitter_stream = Stream(auth, TweetsListener(c_socket), tweet_mode='extended')
    twitter_stream.filter(track=keyword, languages=["en"])


def start_listening_and_send_tweets(topics):
    #init a socket to send the data from twitter to our pyspark context for fault handling
    s = socket.socket()
    host = host_name
    #tweets received are sent on port 5555 when a connection is formed
    port = 5555
    s.bind((host, port))
    s.listen(5)
    c, addr = s.accept()
    sendData(c, topics)
