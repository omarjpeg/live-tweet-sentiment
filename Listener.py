from tweepy import Stream
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import json
import socket

consumer_key = 'AYoR5fpYK5IZuvHJtCBNd7tjx'
consumer_secret = 'XXqvJtQ8HOZyls19SbgWNYQG6MguLDkA8qtMFFrMw1uTA2OWhD'
access_token = '1438526381109833735-mBSov4dGzXLMaaHRtKGJ4xw6O77sGH'
access_secret = 'zD4d9XGCSxvy1uKA5Nv4cWlRcFse9ZKwuzvLrV7oSS9r4'
host_name = socket.gethostbyname(socket.gethostname())


class TweetsListener(StreamListener):
    # tweet object listens for the tweets
    def __init__(self, csocket):
        super().__init__()
        self.client_socket = csocket

    def on_data(self, data):
        try:
            msg = json.loads(data)
            print("New tweet!")
            # if tweet is longer than 140 characters
            if "extended_tweet" in msg:
                # add at the end of each tweet "t_end"
                self.client_socket \
                    .send((str(msg['created_at']) + '__TIME_END__' + str(msg['extended_tweet']['full_text']).replace(
                    "\n", " ") + '\n'
                           ).encode('utf-8'))
            elif "retweeted_status" in msg:
                if "extended_tweet" in msg["retweeted_status"]:
                    self.client_socket \
                    .send((str(msg['retweeted_status']['created_at']) + '__TIME_END__' + str(msg['retweeted_status']['extended_tweet']['full_text']).replace(
                    "\n", " ") + '\n'
                           ).encode('utf-8'))
                else:
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
    print('start sending data from Twitter to socket')
    # authentication based on the credentials
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    # start sending data from the Streaming API
    twitter_stream = Stream(auth, TweetsListener(c_socket), tweet_mode='extended')
    twitter_stream.filter(track=keyword, languages=["en"])


def start_listening_and_send_tweets(topics):
    s = socket.socket()
    host = host_name
    port = 5555
    s.bind((host, port))
    s.listen(5)
    c, addr = s.accept()
    sendData(c, topics)
