import re
import string
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import nltk
import nltk.classify.util
from nltk.corpus import stopwords
nltk.download("stopwords")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import metrics, datasets

from keras import optimizers
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, Embedding, Bidirectional, SimpleRNN, LSTM, Conv1D, MaxPooling1D, Input, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

MAX_NB_WORDS = 11000 # 字典裡總共多少字
MAX_SEQUENCE_LENGTH = 50 # 每句sentence的長度(max in dataset is 94)
EMBEDDING_DIM = 100 # word embedding的維度

def preprocessing(sentences):    
    sentences = sentences.str.replace('[^\w\s]',' ')
    sentences = sentences.str.replace('\d+', ' ')
    '''
    #Remove punctuation from string
    punctuation_replacer = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    sentences = sentences.str.translate(punctuation_replacer)
    '''
    # Remove stopwords
    stopword = stopwords.words('english')
    sentences = sentences.str.lower()
    sentences = sentences.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopword)]))
    
    return sentences

def analysis_data(sentences):
    num_word = sentences.str.split().str.len()
    num_word.plot.hist(bins=20, range=(0,200)).set_xlabel("Number of Word in a Sentence")
    print(num_word.describe())

def split_train_test(path):
    #load training data
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=['reviews.text','reviews.rating'])
    
    df['reviews.text'] = preprocessing(df['reviews.text'])
    
    #analysis_data(df['reviews.text'] )
    
    #print(df['reviews.text'])
    X = np.array(df['reviews.text'])
    y = np.array(df['reviews.rating']).astype(int)
    #y = df['reviews.rating'].apply(np.array)

    # 1~5 轉成 0~4
    y = y - 1
    #print(type(y))
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    
    train_data = np.transpose(np.vstack((X_train, Y_train)))
    test_data = np.transpose(np.vstack((X_test, Y_test)))
    
    train_data_df = pd.DataFrame(train_data)
    train_data_df.to_csv("./split_data/train_data.csv", index=False, header=['reviews.text','reviews.rating'])
    
    test_data_df = pd.DataFrame(test_data)
    test_data_df.to_csv("./split_data/test_data.csv", index=False, header=['reviews.text','reviews.rating'])
    
def load_data():
    train_data = pd.read_csv('./split_data/train_data.csv', low_memory=False)
    X_train = np.array(train_data['reviews.text']).astype(str)
    Y_train = np.array(train_data['reviews.rating']).astype(int)
    Y_train = to_categorical(np.asarray(Y_train, dtype=np.int32))
    
    test_data = pd.read_csv('./split_data/test_data.csv', low_memory=False)
    X_test = np.array(test_data['reviews.text']).astype(str)
    Y_test = np.array(test_data['reviews.rating']).astype(int)
   
    return X_train, X_test, Y_train, Y_test

def pretrain_word_embedding(X_train, X_test):    
    sentences = []
    total_num_token = 0
    for sent_str in X_train:
        tokens = nltk.word_tokenize(sent_str.lower())
        #total_num_token += len(tokens)
        #print('token:', tokens)
        sentences.append(tokens)
    #print('sentence average length:', total_num_token/len(sentences))
    
    from gensim.models import FastText
    fast_text_embeddings = FastText(sentences, size=EMBEDDING_DIM, window=5, min_count=3, workers=4, sg=1)  
    
    '''
    # calculate max number of words in a sentence
    max_len = 0
    for sentence in X_train:      
        #print(sentence)
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > max_len:
            MAX_SEQUENCE_LENGTH = len(words)
    print('MAX_SEQUENCE_LENGTH: ', MAX_SEQUENCE_LENGTH)
    '''
    # 將訓練資料的單字轉成向量
    texts = X_train
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    train_sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    X_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # 將測試資料的單字轉成向量
    test_sequences = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    '''
    with open('params/word_index.txt', 'w') as txt_file:
        for word, i in word_index.items():
            txt_file.write(str(i)+' : '+ word +'\n')    
    '''
          
    # 轉成 Embedding 層的 input vector
    num_unk = 0 # <unk>的數量
    total_num_index = min(MAX_NB_WORDS, len(word_index)+1)
    embedding_matrix = np.zeros((total_num_index, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        #print(word)
        try:
            embedding_vector = fast_text_embeddings[word]
            embedding_matrix[i] = embedding_vector           
        except KeyError:# words not found in embedding index will be all-zeros.
            num_unk += 1
        '''
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            #print(word)
            num_unk += 1
        '''    

    print('NUM_UNK:', num_unk)   
    
    return X_train, X_test, total_num_index, embedding_matrix

def create_dictionary(X_train, X_test):
     # Word mapping, 0:padding, 1:<unk>
    word_pop = {}
    test_dic = {}
    test_unk_dic = {}
    
    word_mapping = {}
    word_indexing = 2
    word_freq_threshold = 1
    # Zero Padding to max_len
    max_sequence_len = MAX_SEQUENCE_LENGTH   
      
     # Culculate each word's frequency
    for sentence in X_train:         
        for word in sentence.split(' '):
            if word not in word_pop:
                word_pop[word] = 1
            else:
                word_pop[word] += 1
     
        '''
         # Find max word count in a sentence
        words = nltk.word_tokenize(sentence.lower())
        if len(words) >= max_len:
            max_sequence_len = len(words)
        ''' 
    # Map words with frequency > threshold to index, otherwise to 1
    for k in sorted(word_pop.keys()):
        if word_pop[k] >= word_freq_threshold:
            word_mapping[k] = word_indexing
            word_indexing += 1
        else:
            word_mapping[k] = 1
    total_num_index = word_indexing
    print('Total',total_num_index,'words mapped into index')
    
    # Save word mapping
    '''
    with open('params/word_mapping.txt', 'w') as txt_file:
        for key in word_mapping:
            txt_file.write('%s %s\n' % (key, word_mapping[key]))
    '''
    
    # Transform train sentences into sequence of index
    mapped_X_train = []
    for sentence in X_train:
        tmp = []
        for word in sentence.split(' '):
            tmp.append(word_mapping[word])
        if len(tmp)<max_sequence_len:
            tmp.extend([0]*(max_sequence_len-len(tmp)))
        elif len(tmp) > max_sequence_len:
            tmp = tmp[:max_sequence_len]
        mapped_X_train.append(tmp)   
    X_train = np.array(mapped_X_train)
    
    # Transform test sentences into sequence of index
    mapped_X_test = []
    for sentence in X_test:
        tmp = []
        for word in sentence.split(' '):
            if word not in word_mapping:
                test_unk_dic[word]=1
                tmp.append(1)
            else:
                tmp.append(word_mapping[word])
                test_dic[word]=1
        if len(tmp)<max_sequence_len:
            tmp.extend([0]*(max_sequence_len-len(tmp)))
        elif len(tmp) > max_sequence_len:
            tmp = tmp[:max_sequence_len]
        mapped_X_test.append(tmp)
    X_test = np.array(mapped_X_test)
    
    print("Total",  len(test_dic), "words in test data")
    print(len(test_unk_dic), "words in test data didn't be coverd")
    print("%.2f%% of words in test data didn't be coverd" % (len(test_unk_dic)/len(test_dic)*100) )
    
    return (X_train, X_test, total_num_index)

def text_cnn(maxlen, embedding_dim, total_num_index, embedding_matrix):
    # Inputs
    comment_seq = Input(shape=[maxlen], name='x_seq')

    # Word embedding layer
    if embedding_matrix is not None:
        emb_comment = Embedding(total_num_index, embedding_dim, weights=[embedding_matrix], trainable=False)(comment_seq)
    else:
        emb_comment = Embedding(total_num_index, embedding_dim, trainable=True)(comment_seq)
#     # Embeddings layers
#     emb_comment = Embedding(max_features+1, embed_size)(comment_seq)

    # conv layers
    convs = []
    filter_sizes = [3, 4]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)

    output = Dense(units=5, activation='softmax')(output)

    model = Model([comment_seq], output)
    #     adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model

def text_cnn_lstm(maxlen, embedding_dim, total_num_index, embedding_matrix):
    model = Sequential()

    # Word embedding layer
    if embedding_matrix is not None:
        model.add(Embedding(total_num_index, embedding_dim, weights=[embedding_matrix], trainable=False))  
    else:
        model.add(Embedding(total_num_index,embedding_dim , trainable=True))
    fsz = 3
    model.add(Conv1D(filters=100, kernel_size=fsz, activation='relu'))
    model.add(MaxPooling1D(maxlen - fsz + 1))
    
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(units=5, activation='softmax'))
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model

def text_lstm(embedding_dim, total_num_index, embedding_matrix):
    model = Sequential()

    # Word embedding layer
    if embedding_matrix is not None:
        model.add(Embedding(total_num_index, embedding_dim, weights=[embedding_matrix], trainable=False))  
    else:
        model.add(Embedding(total_num_index,embedding_dim , trainable=True))
    
    model.add(LSTM(8, return_sequences=True))
    model.add(Bidirectional(LSTM(8)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(units=5, activation='softmax'))
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model

def infer(model, X_test, Y_test):
    # Testing
    #model = load_model('params/rnn.model')

    probabilities = model.predict(X_test)
    
    predictions=[]
   
    for i in range(len(probabilities)):
        pred = probabilities[i]
        pred = np.argmax(pred)  #select each result's maximum index 
        predictions.append(pred)
    predictions=np.array(predictions)

    accuracy = np.mean(predictions == Y_test)
    print("Prediction Accuracy: %.2f%%" % (accuracy*100))
    plot_confusion_matrix(Y_test, predictions)

def plot_confusion_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index = [i+1 for i in range(5)], columns = [i+1 for i in range(5)])
    plt.figure(figsize = (10,7))

    ax = sn.heatmap(df_cm,  annot=True, fmt ='g' ,cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

if __name__ == '__main__':
    # Data preprocessing
    X_train, X_test, Y_train, Y_test = load_data()
    
    # Use pretrain word embedding
    X_train, X_test, total_num_index, embedding_matrix = pretrain_word_embedding(X_train, X_test)
    
    # LSTM model
    lstm_model = text_lstm(EMBEDDING_DIM, total_num_index, embedding_matrix)
    # Set check point to early stop and save the best model
    check = ModelCheckpoint('params/lstm.model', monitor='val_acc', verbose=0, save_best_only=True)
    lstm_model.fit(X_train, Y_train, batch_size=128, epochs=100, callbacks=[check, EarlyStopping(patience=5)], validation_split=0.1)