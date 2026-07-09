import re
import os
import numpy as np
from src.logger import logging
import pandas as pd
import tensorflow_hub as hub
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from pickle import dump, load
from tqdm import tqdm
from tensorflow import keras
import json

image_path = "flickr30k/flickr30k_images"
def load_caption(cap_path:str) -> pd.DataFrame:
    """Load image paths into a DataFrame."""
    try:
        caption = pd.read_csv(cap_path)
        logging.info("Captions loaded successfully.")
        return caption
    except Exception as e:
        logging.error(f"Error occurred while loading captions: {e}")
        raise

MAX_LENGTH = 40
VOCAB_SIZE = 10000

# loading vit model 
def load_vit_model():
    """load vit model from tfhub"""
    try:
        vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
        vit_model = hub.KerasLayer(vit_url, trainable=False, input_shape=(224, 224, 3))
        logging.info("VIT model loaded successfully from TensorFlow Hub.")
        return vit_model
    except Exception as e:
        logging.error(f"Error occurred while loading VIT model: {e}")
        raise  


# cleaning caption
def clean_caption(caption:pd.Series) -> pd.Series:
    """Cleans the caption by removing unwanted characters and formatting it."""
    try:
        cap = caption.lower().strip()
        if cap.startswith("startseq") and cap.endswith("endseq"):
            mid = cap[len("startseq"):-len("endseq")].strip()
        else:
            mid = cap
        mid = re.sub(r'[^a-z\s]', '', mid)
        mid = re.sub(r'\s+', ' ', mid).strip()
        logging.info("Cleaned caption created")
        return f"startseq {mid} endseq"
    except Exception as e:
        logging.error(f"Error occurred while cleaning caption: {e}")
        raise

# tokenization
def tokenize_captions(caption: pd.Series) -> tuple:
    """Tokenizes the captions and returns the tokenizer, vocabulary size, and maximum caption length."""
    try:
        captions_list = caption['comment'].tolist()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(captions_list)
        with open("data/tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        VOCAB_SIZE = len(tokenizer.word_index) + 1
        MAX_LENGTH = min(40, max(len(c.split()) for c in captions_list))
        with open("data/config.json", "w") as f:
            json.dump({"VOCAB_SIZE": VOCAB_SIZE, "MAX_LENGTH": MAX_LENGTH}, f)
        logging.info("Tokenization completed successfully.")
        logging.info(f"Vocab size: {VOCAB_SIZE} \n Max Caption length: {MAX_LENGTH}")
        return tokenizer, VOCAB_SIZE, MAX_LENGTH
    except Exception as e:
        logging.error(f"Error occurred while tokenizing captions: {e}")
        raise

#splitting the dataset into train, validation and test sets
def split_dataset(caption: pd.DataFrame) -> tuple:
    """Splits the dataset into training, validation, and test sets."""
    try:
        image_ids = caption['image_name'].unique().tolist()
        train_ids, test_ids = train_test_split(image_ids, test_size=0.1, random_state=42)
        train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42)

        train_df = caption[caption['image_name'].isin(train_ids)].reset_index(drop=True)
        val_df   = caption[caption['image_name'].isin(val_ids)].reset_index(drop=True)
        test_df  = caption[caption['image_name'].isin(test_ids)].reset_index(drop=True)

        logging.info("Dataset split into train, validation, and test sets successfully.")
        return train_df, val_df, test_df
    except Exception as e:
        logging.error(f"Error occurred while splitting dataset: {e}")
        raise

#extracting features from images using VIT model
def extract_features(df: pd.DataFrame, image_dir: str)-> dict:
    """Extracts features from images using the VIT model."""
    try:
        vit_model = load_vit_model()
        logging.info("Extracting features from images using VIT model.")
        features = {}
        for img_name in tqdm(df['image_name'].unique()):
            img_path = os.path.join(image_dir, img_name)
            img = keras.preprocessing.image.load_img(img_path, target_size=(224,224))
            img = keras.preprocessing.image.img_to_array(img) / 255.0
            img = np.expand_dims(img, 0)
            feat = vit_model(img)  # (1, 768)
            features[img_name] = feat.numpy()[0]  # (768,)
        with open("data/features.pkl", "wb") as f:
            pickle.dump(features, f)
        logging.info("Feature extraction completed successfully.")
        return features
    except Exception as e:
        logging.error(f"Error occurred while extracting features: {e}")
        raise