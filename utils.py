from nltk.tokenize import TweetTokenizer
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from emosent import get_emoji_sentiment_rank


def create_tweet_vectors(tweets, phrase_model, train_ratio, with_sentiment):
    # create vectors from input data and vector-models
    # returns lists of train and test vectors and labels

    train_vectors = []
    train_labels = []
    valid_vectors = []
    valid_labels = []
    test_vectors = []
    test_labels = []
    
    tweet_count = len(tweets)
    train_limit = int(train_ratio * tweet_count)
    valid_limit = train_limit + int((1-train_ratio)/2 * tweet_count)
    
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    for i in range(0, train_limit):    
        tokens = tokenizer.tokenize(tweets['Text'][i])
        if with_sentiment:
            train_vectors.append(np.append(np.sum([phrase_model[x] for x in tokens], axis=0) / len(tokens),
                                           tweets['Sentiment'][i]))
        else:
            train_vectors.append(np.sum([phrase_model[x] for x in tokens], axis=0) / len(tokens))
        train_labels.append(tweets['Label'][i])
        
    for i in range(train_limit, valid_limit):
        tokens = tokenizer.tokenize(tweets['Text'][i])
        if with_sentiment:
            valid_vectors.append(np.append(np.sum([phrase_model[x] for x in tokens], axis=0) / len(tokens),
                                           tweets['Sentiment'][i]))
        else:
            valid_vectors.append(np.sum([phrase_model[x] for x in tokens], axis=0) / len(tokens))
        valid_labels.append(tweets['Label'][i])
        
    for i in range(valid_limit, tweet_count):
        tokens = tokenizer.tokenize(tweets['Text'][i])
        if with_sentiment:
            test_vectors.append(np.append(np.sum([phrase_model[x] for x in tokens], axis=0) / len(tokens),tweets['Sentiment'][i]))
        else:
            test_vectors.append(np.sum([phrase_model[x] for x in tokens], axis=0) / len(tokens))
        test_labels.append(tweets['Label'][i])

    return train_vectors, train_labels, valid_vectors, valid_labels, test_vectors, test_labels


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

    tweets.loc[emoji_list].reset_index(drop=True).to_csv(path, index=False)
    
    
def create_emoji_sentiment(tweets, emoji_model):
    # create new column with emoji sentiment
    
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    tweets['Sentiment'] = 0
    for i in range (0, len(tweets)):
        sentiment = 0
        tokens = tokenizer.tokenize(tweets['Text'][i])
        for token in tokens:
            if token in emoji_model:
                try:
                    sentiment += get_emoji_sentiment_rank(token)["sentiment_score"]
                except:
                    pass
        tweets.loc[i, 'Sentiment'] = sentiment
        
    return tweets
                 
    
def encode_data_get_embeddings(tweets, train_ratio, w2v, e2v):
    tw_tok = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    
    tweets = tweets.sample(frac=1).reset_index(drop=True)
    
    # text reprocessing
    tweets['Text'] = tweets['Text'].apply(lambda x: ' '.join([w for w in tw_tok.tokenize(x)]))
#     tweets['Text'] = tweets['Text'].apply(lambda x:  re.sub('#[a-zA-Z]* ','', x))  odstranenie hashtagov
    x = np.array(tweets['Text'])

    # one-hot encoding of label
    y = np.array(tweets['Label'].apply(lambda x: 1 if x == 'Positive' else (0 if x =='Neutral' else 2)))
#     y = np.array(tweets['Label'].apply(lambda x: 1 if x == 'sad' else (0 if x =='others' else (2 if x =='happy' else 3))))

    y = to_categorical(y)

    tweet_count = len(tweets)
    limit = int(train_ratio * tweet_count)
    
    train_x = x[:limit]
    train_y = y[:limit]
    test_x = x[limit:]
    test_y = y[limit:]
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_x)
    
    train_tweets = train_x
    test_tweets = train_x

    train_x = tokenizer.texts_to_sequences(train_x)
    test_x = tokenizer.texts_to_sequences(test_x)
    
    train_x = pad_sequences(train_x, maxlen=79)
    test_x = pad_sequences(test_x,  maxlen=79)
    
    sent_arr = np.zeros((len(train_x),1))   # create array of zeros
    for i in range(0, len(train_x)):
        sent = 0
        l = 0
        for char in train_tweets[i]:
            try:
                sent += get_emoji_sentiment_rank(char)["sentiment_score"] 
                l+=1
            except:
                pass
        if l>0:
            sent /= l    
        sent_arr[i] = sent+1
    train_x = np.append(train_x, sent_arr, axis=1)
    
    sent_arr = np.zeros((len(test_x),1))   # create array of zeros
    for i in range(0, len(test_x)):
        sent = 0
        l = 0
        for char in test_tweets[i]:
            try:
                sent += get_emoji_sentiment_rank(char)["sentiment_score"] 
                l+=1
            except:
                pass
        if l>0:
            sent /= l
        sent_arr[i] = sent+1
    test_x = np.append(test_x, sent_arr, axis=1)
    
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

    vocab_size = len(tokenizer.word_index) + 1
            
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    
    return train_x, train_y, test_x, test_y, embedding_matrix


    iter = 0
    with io.open(emoji2vec_file, encoding="utf8") as f:
        for line in f:
            if iter == 0:
                iter += 1
                continue
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector 
            
    return embeddingsIndex, 300
