import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, df, features, tokenizer, max_length, batch_size=32, shuffle=True,**kwargs):
        super().__init__(**kwargs)
        self.df = df.copy().reset_index(drop=True)
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, idx):
        batch_indices = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = self.df.iloc[batch_indices]
        
        x_img, x_seq, y = [], [], []
        for _, row in batch.iterrows():
            img_feature = self.features[row['image_name']]  
            seq = self.tokenizer.texts_to_sequences([row['comment']])[0]
            seq = seq[:self.max_length]  
            seq_pad = pad_sequences([seq], maxlen=self.max_length, padding='post')[0]
            
            x_img.append(img_feature)
            # caption input
            x_seq.append(seq_pad[:-1])
            # target output
            y.append(seq_pad[1:])      
        
        x_img = np.array(x_img, dtype=np.float32)
        x_seq = np.array(x_seq, dtype=np.int32)
        y     = np.array(y,     dtype=np.int32)
        return (x_img, x_seq), y
