IMDB = "imdb"
AGNEWS = "ag_news"
TWEETEVAL_SENTIMENT = "tweeteval_sentiment"
TWEETEVAL_OFFENSIVE = "tweeteval_offensive"
DICT_IMDB = {'0' : 'negative' , '1' : 'positive'}
DICT_AGNEWS = {'0' : 'World' , '1' : 'Sports' , '2' : 'Business', '3' : 'SciTech'}
DICT_TWEETEVAL_SENTIMENT = {'0': 'negative','1': 'neutral', '2' : 'positive'}
DICT_TWEETEVAL_OFFENSIVE = {'0': 'non-offensive','1': 'offensive'}

DICT_DATASET_ALIAS = {IMDB : "imdb", AGNEWS : "ag_news", TWEETEVAL_SENTIMENT : "cardiffnlp/tweet_eval-sentiment", TWEETEVAL_OFFENSIVE : "cardiffnlp/tweet_eval-offensive"}

DICT_CATEGORIES = { IMDB : DICT_IMDB , AGNEWS : DICT_AGNEWS, TWEETEVAL_SENTIMENT: DICT_TWEETEVAL_SENTIMENT, TWEETEVAL_OFFENSIVE: DICT_TWEETEVAL_OFFENSIVE}