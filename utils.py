from nltk.tokenize import TweetTokenizer
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from emosent import get_emoji_sentiment_rank


def create_tweet_vectors(tweets, phrase_model, train_ratio, with_sentiment):
    # create vectors from input data and vector-models
    # returns lists of train and test vectors and labels

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    train_vectors = []
    train_labels = []
    test_vectors = []
    test_labels = []
    
    tweet_count = len(tweets)
    limit = int(train_ratio * tweet_count)

    for i in range(0,limit):    
        tokens = tokenizer.tokenize(tweets['Text'][i])
        if with_sentiment:
            train_vectors.append(np.append(np.sum([phrase_model[x] for x in tokens], axis=0) / len(tokens),tweets['Sentiment'][i]))
        else:
            train_vectors.append(np.sum([phrase_model[x] for x in tokens], axis=0) / len(tokens))
        train_labels.append(tweets['Label'][i])
        
    for i in range(limit, tweet_count):
        tokens = tokenizer.tokenize(tweets['Text'][i])
        if with_sentiment:
            test_vectors.append(np.append(np.sum([phrase_model[x] for x in tokens], axis=0) / len(tokens),tweets['Sentiment'][i]))
        else:
            test_vectors.append(np.sum([phrase_model[x] for x in tokens], axis=0) / len(tokens))
        test_labels.append(tweets['Label'][i])

    return train_vectors, train_labels, test_vectors, test_labels


def create_emoji_tweets(tweets, emoji_model, path):
    # create emoji-tweets from input data and emoji-model
    # save dataframe of tweets, which contains at least 1 emoji
    
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    emoji_list = []
    for i in range (0, len(tweets)):
        tokens = tokenizer.tokenize(tweets['Text'][i])
        for token in tokens:
            if token in emoji_model:
                emoji_list.append(i)
                break

    tweets.loc[emoji_list].reset_index(drop=True).to_csv(path)
    
    
def create_emoji_sentiment(tweets, emoji_model):
    # create new column with emoji sentiment
    
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    tweets['Sentiment'] = 0
    
    for i in range (0, len(tweets)):
        sentiment = 0
        tokens = tokenizer.tokenize(tweets['Text'][i])
        for token in tokens:
            if token in emoji_model:
                sentiment += get_emoji_sentiment_rank(token)["sentiment_score"]
        tweets.loc[i, 'Sentiment'] = sentiment
        
    return tweets
                 
    
def encode_data_get_embeddings(tweets, train_ratio, w2v, e2v):
    tw_tok = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    
    tweets = tweets.sample(frac=1).reset_index(drop=True)
    
    # Preprocessing textu
    tweets['Text'] = tweets['Text'].apply(lambda x: ' '.join([w for w in tw_tok.tokenize(x)]))
#     tweets['Text'] = tweets['Text'].apply(lambda x:  re.sub('#[a-zA-Z]* ','', x))  odstranenie hashtagov
    x = np.array(tweets['Text'])

    # One-hot encoding labelu
    y = np.array(tweets['Label'].apply(lambda x: 1 if x == 'Positive' else (0 if x =='Neutral' else -1)))
#     y = np.array(tweets['Label'].apply(lambda x: 1 if x == 'Positive' else 0))

    tweet_count = len(tweets)
    limit = int(train_ratio * tweet_count)
    
    train_x = x[:limit]
    train_y = y[:limit]
    test_x = x[limit:]
    test_y = y[limit:]
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_x)

    train_x = tokenizer.texts_to_sequences(train_x)
    test_x = tokenizer.texts_to_sequences(test_x)

    vocab_size = len(tokenizer.word_index) + 1

    train_x = pad_sequences(train_x, padding='post', maxlen=80)
    test_x = pad_sequences(test_x, padding='post', maxlen=80)
    
    embeddings_dictionary = dict()

    for line in w2v:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    
    if e2v is not None:
        for line in e2v:
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions

    embedding_matrix = np.zeros((vocab_size, 300))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    
    return train_x, train_y, test_x, test_y, embedding_matrix









