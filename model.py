import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, GRU, LSTM, Bidirectional, Dropout, Attention, Conv1D, GlobalMaxPooling1D, Flatten


class BaselineModel(keras.Model):
    
    def __init__(self, embedding_matrix):
        super(BaselineModel, self).__init__()
        self.embe = Embedding(embedding_matrix.shape[0],
                              300,
                              weights=[embedding_matrix],
                              input_length=80,
                              trainable=False)
        self.simple = SimpleRNN(128) 
        self.dense = Dense(units=1, activation='sigmoid')  
        
    def call(self, x):  
        x = self.embe(x)
        x = self.simple(x)  
        x = self.dense(x) 
        return x
    
    
class BaselineDropModel(keras.Model):
    
    def __init__(self, embedding_matrix):
        super(BaselineDropModel, self).__init__()
        self.embe = Embedding(embedding_matrix.shape[0],
                              300,
                              weights=[embedding_matrix],
                              input_length=80,
                              trainable=False)
        self.drop = Dropout(0.5) 
        self.simple = SimpleRNN(128) 
        self.dense = Dense(units=1, activation='sigmoid')  
        
    def call(self, x):  
        x = self.embe(x)
        x = self.drop(x)
        x = self.simple(x)  
        x = self.dense(x) 
        return x
    
    
class LstmModel(keras.Model):
    
    def __init__(self, embedding_matrix):
        super(LstmModel, self).__init__()
        self.embe = Embedding(embedding_matrix.shape[0],
                              300,
                              weights=[embedding_matrix],
                              input_length=80,
                              trainable=False)
        self.lstm = LSTM(128, return_sequences=True)
#         self.attent = Attention(name='attention_weight')
        self.dense = Dense(units=1, activation='sigmoid')  
        
    def call(self, x):  
        x = self.embe(x)
        x = self.lstm(x) 
        x = self.dense(x) 
        return x

    
class BiLstmModel(keras.Model):
    
    def __init__(self, embedding_matrix):
        super(BiLstmModel, self).__init__()
        self.embe = Embedding(embedding_matrix.shape[0],
                              300,
                              weights=[embedding_matrix],
                              input_length=80,
                              trainable=False)
        self.lstm = Bidirectional(LSTM(256, return_sequences=True))
        self.bilstm = Bidirectional(LSTM(256))
        self.dense = Dense(units=3, activation='softmax')  
#         self.dense = Dense(units=1, activation='sigmoid')  
        
    def call(self, x):  
        x = self.embe(x)
        x = self.bilstm(x) 
        x = self.dense(x) 
        return x
    
    
class TestModel(keras.Model):
    
    def __init__(self, embedding_matrix):
        super(TestModel, self).__init__()
        self.embe = Embedding(embedding_matrix.shape[0],
                              300,
                              weights=[embedding_matrix],
                              input_length=80,
                              trainable=False)
        self.conv = Conv1D(300, 3, padding="same", activation="relu")
        self.pool = GlobalMaxPooling1D()
        self.drop = Dropout(0.4)
        self.lstm = LSTM(50, return_sequences=True)
        self.drop1 = Dropout(0.25)
        self.flat = Flatten()
        self.dense1 = Dense(units=128, activation='relu')  
        self.drop2 = Dropout(0.45)
        self.dense2 = Dense(units=4, activation='softmax') 
        
    def call(self, x):  
        x = self.embe(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.lstm(x)
        x = self.drop1(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.drop2(x) 
        x = self.dense2(x) 
        return x
    
    
class CombiModel(keras.Model):
    
    def __init__(self, embedding_matrix, bi_units, ls_units):
        super(CombiModel, self).__init__()
        self.embe = Embedding(embedding_matrix.shape[0],
                              300,
                              weights=[embedding_matrix],
                              input_length=80,
                              trainable=False)
        self.bilstm = Bidirectional(LSTM(256, return_sequences=True))
        self.lstm = LSTM(256)
        self.dense = Dense(units=1, activation='sigmoid')  
        
    def call(self, x):  
        x = self.embe(x)
        x = self.bilstm(x)
        x = self.lstm(x)
        x = self.dense(x) 
        return x
    
    
    
class ConvModel(keras.Model):
    
    def __init__(self, embedding_matrix):
        super(ConvModel, self).__init__()
        self.embe = Embedding(embedding_matrix.shape[0],
                              300,
                              weights=[embedding_matrix],
                              input_length=80,
                              trainable=False)
        self.drop = Dropout(0.5) 
        self.conv = Conv1D(300, 7, padding="valid", activation="relu", strides=3)
        self.pool = GlobalMaxPooling1D()
        self.van_dense = Dense(300, activation="relu")
        self.dense = Dense(units=1, activation='sigmoid')  
        
    def call(self, x):  
        x = self.embe(x)
        x = self.drop(x)
        
        x = self.conv(x)  
        x = self.conv(x) 
        x = self.pool(x) 
        
        x = self.van_dense(x)
        x = self.drop(x)
        
        x = self.dense(x) 
        return x
    
    
    
    