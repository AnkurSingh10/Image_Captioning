import os
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from models.model_testing import generate_caption
import json
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from layers.Custom_layer_model import Transformer_decoder, Expand_Dimension, Masked_Loss, PositionalEmbedding
from tensorflow import keras
import textwrap
from models.feature_engineering import load_vit_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import load
import random
from tqdm import tqdm


model = load_model("caption_model.h5",
                   custom_objects={"Transformer_decoder": Transformer_decoder,
                                   "PositionalEmbedding": PositionalEmbedding,
                                   "Masked_Loss": Masked_Loss})

test_df = pd.read_csv("data/test.csv")

with open("data/tokenizer.pkl", "rb") as f:
    tokenizer = load(f)

with open("data/features.pkl", "rb") as f:
    features = load(f)

with open("data/config.json", "r") as f:
    config = json.load(f)

VOCAB_SIZE = config["VOCAB_SIZE"]
MAX_LENGTH = config["MAX_LENGTH"]

def clean_tokens(tokens):
    return [
        t for t in tokens
        if t not in {"startseq", "endseq", "<pad>"}
    ]

test_images = test_df['image_name'].unique().tolist()
random.seed(42)
sample_images = random.sample(test_images, min(500, len(test_images)))

refs, hyps = [], []
smooth = SmoothingFunction().method1

for img_id in tqdm(sample_images):

    img_loc = "flickr30k_images" + "/"+img_id
    pred_caption = generate_caption(
        img_loc,
        model,
        tokenizer,
        MAX_LENGTH - 1       
    )

    hyp_tokens = clean_tokens(pred_caption.split())
    hyps.append(hyp_tokens)

    true_caps = test_df[test_df['image_name'] == img_id]['comment'].tolist()
    ref_tokens = [clean_tokens(cap.split()) for cap in true_caps]

    refs.append(ref_tokens)

print("BLEU-1:", corpus_bleu(refs, hyps, weights=(1,0,0,0), smoothing_function=smooth))
print("BLEU-2:", corpus_bleu(refs, hyps, weights=(0.5,0.5,0,0), smoothing_function=smooth))
print("BLEU-3:", corpus_bleu(refs, hyps, weights=(0.33,0.33,0.33,0), smoothing_function=smooth))
print("BLEU-4:", corpus_bleu(refs, hyps, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth))