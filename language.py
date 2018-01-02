import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.fftpack import fft
import librosa


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
import tensorflow as tf
from keras import optimizers

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,file_ext="*.wav"):
    
    features=[]
    for fn in glob.glob(os.path.join(parent_dir, "train", file_ext)):
        
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
        except Exception as e:
            print("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features.append(ext_features)
        #labels = np.append(labels, fn.split('/')[2].split('-')[1])
    return np.array(features)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def split(df):
    return (df.iloc[:,0:len(df.columns)-1]),df.iloc[:,-1]

#Try PCA

def perform_pca(train_data,test_data):
    X_std = StandardScaler().fit_transform(train_data)
    
    pca = PCA(n_components=100)
    pca.fit_transform(train_data)
    #print("Variance expalined by different components in training data \n", pca.explained_variance_ratio_)
    #Explained variance
    
    pca = PCA().fit(X_std)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
    
    pca1 = PCA(n_components=100)
    pca1.fit_transform(train_data)
    #print("Variance expalined by different components in training data \n", pca.explained_variance_ratio_)
   
    reduced_train_data=pca1.transform(train_data)
    
    
    reduced_test_data=pca1.transform(test_data)
    
    return reduced_train_data, reduced_test_data

def train_model(df_train,train_y):

    num_labels = train_y.shape[1]
    filter_size = 2

    #build model
    model = Sequential()

    model.add(Dense(128, input_shape=(100,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('sigmoid'))

    #sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    #Training the model
    model.fit(np.array(df_train), np.array(train_y), batch_size=25, epochs=50)
    return model
    

def main():
    parent_dir = os.getcwd()
    tr_features = parse_audio_files(parent_dir)

    train_y=[]
    with open('train-y') as f:
        train_y.append([line.strip() for line in f])

    data=pd.DataFrame(tr_features)
    data['y']= train_y[0]
    print(data)

    #np.random.seed(9001)
    dev_data=data.sample(frac=0.2)
    train_data=data.drop(dev_data.index)
    train_data.reset_index(inplace=True,drop=True)
    dev_data.reset_index(inplace=True,drop=True)
  
    train_x,train_y=split(train_data)
    dev_x,dev_y=split(dev_data)
    dev_y=np_utils.to_categorical(dev_y, 2)
    train_y=np_utils.to_categorical(train_y, 2)

    df_train,df_dev= perform_pca(train_x,dev_x)    

    network=train_model(df_train,train_y)
    
    score = network.evaluate(np.array(df_dev), np.array(dev_y), verbose=0)
    print(score)






if __name__=="__main__":
    main()