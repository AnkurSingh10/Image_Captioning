import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow import keras
import textwrap
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import load

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.Custom_layer_model import TransformerDecoder, ExpandDims, masked_loss, PositionalEmbedding
from src.models.feature_engineering import load_vit_model

vit_model = load_vit_model()
model = load_model(
    PROJECT_ROOT / "data" / "caption_model.h5",
    custom_objects={
        "TransformerDecoder": TransformerDecoder,
        "ExpandDims": ExpandDims ,
        "PositionalEmbedding": PositionalEmbedding,
        "masked_loss": masked_loss ,
    },
)

test_df = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")

with open(PROJECT_ROOT / "data" / "tokenizer.pkl", "rb") as f:
    tokenizer = load(f)

with open(PROJECT_ROOT / "data" / "features.pkl", "rb") as f:
    features = load(f)

with open(PROJECT_ROOT / "data" / "config.json", "r") as f:
    config = json.load(f)

VOCAB_SIZE = config["VOCAB_SIZE"]
MAX_LENGTH = config["MAX_LENGTH"]

def generate_caption(image_path, model, tokenizer, max_length):
    img = keras.preprocessing.image.load_img(image_path, target_size=(224,224))
    img = keras.preprocessing.image.img_to_array(img) / 255.0
    img = np.expand_dims(img, 0)
    feature = vit_model(img)[0].numpy()    
    caption = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding='post')
        y_pred = model.predict([feature.reshape(1, -1), seq], verbose=0)
        next_index = np.argmax(y_pred[0, len(caption.split())-1])
        next_word = tokenizer.index_word.get(next_index, '')
        if next_word == '' or next_word == 'endseq':
            break
        caption += ' ' + next_word
    return caption

# image_url = PROJECT_ROOT / "data" / "img2.png"
# new_caption = generate_caption(image_url, model, tokenizer,39 )
# # new_caption = generate_caption("/kaggle/input/tttttttt/OIP (1).webp", model, tokenizer, max_len_gen)
# print(new_caption.replace("startseq", "").replace("endseq", "").strip())
# img = Image.open(image_url)
# plt.imshow(img)
def show_prediction(
    image_path,
    model,
    tokenizer,
    max_length,
    dataframe,
    save_image = False
):
  
    image_id =os.path.basename(image_path)
    # print(image_id)

    # Generate caption 
    pred_caption = generate_caption(
        image_path,
        model,
        tokenizer,
        max_length
    )

    pred_caption = (
        pred_caption
        .replace("startseq", "")
        .replace("endseq", "")
        .strip()
    )

    true_caption = dataframe[dataframe["image_name"] == image_id]["comment"].iloc[1]
    true_caption = (
        true_caption
        .replace("startseq", "")
        .replace("endseq", "")
        .strip()
    )

    img = Image.open(image_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")

    title_text = (
         "\n\nActual: \n"
        +"\n".join(textwrap.wrap(true_caption, 60))
        +"\n"
        +"Predicted:\n"
        + "\n".join(textwrap.wrap(pred_caption, 60))
    )

    plt.title(title_text, fontsize=11)
    if save_image:
        save_path = os.path.join(f"{image_id}_caption.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Image saved at: {save_path}")    
    plt.show()


image_url = PROJECT_ROOT / "data" / "Screenshot 2026-07-09 151414.png"
show_prediction(
    image_url,
    model,
    tokenizer,
    MAX_LENGTH - 1,
    test_df,
    save_image = True
)