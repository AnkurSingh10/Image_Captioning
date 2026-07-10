import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.Custom_layer_model import ExpandDims, PositionalEmbedding, TransformerDecoder, masked_loss
from src.models.feature_engineering import load_vit_model

app = FastAPI(title="Image Captioning API")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))
uploads_dir = PROJECT_ROOT / "app" / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")


@lru_cache(maxsize=1)
def get_vit_model():
    return load_vit_model()


@lru_cache(maxsize=1)
def get_model_bundle():
    with open(PROJECT_ROOT / "data" / "tokenizer.pkl", "rb") as f:
        tokenizer = load(f)

    with open(PROJECT_ROOT / "data" / "config.json", "r") as f:
        config = json.load(f)

    model = load_model(
        PROJECT_ROOT / "data" / "caption_model.h5",
        custom_objects={
            "TransformerDecoder": TransformerDecoder,
            "ExpandDims": ExpandDims,
            "PositionalEmbedding": PositionalEmbedding,
            "masked_loss": masked_loss,
        },
        compile=False,
    )
    return model, tokenizer, config["MAX_LENGTH"]


def preprocess_image(image_path: Path):
    image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    array = keras.preprocessing.image.img_to_array(image) / 255.0
    return np.expand_dims(array, 0)


def generate_caption(image_path: Path, model, tokenizer, max_length: int) -> str:
    image_batch = preprocess_image(image_path)
    feature_vector = get_vit_model()(image_batch)[0].numpy()

    caption = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
        prediction = model.predict([feature_vector.reshape(1, -1), sequence], verbose=0)
        next_index = int(np.argmax(prediction[0, len(caption.split()) - 1]))
        next_word = tokenizer.index_word.get(next_index, "")
        if not next_word or next_word == "endseq":
            break
        caption += " " + next_word

    return caption.replace("startseq", "").replace("endseq", "").strip()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"caption": None, "image_name": None, "image_url": None, "error": None},
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"caption": None, "image_name": None, "image_url": None, "error": "Please upload an image file."},
        )

    file_path = uploads_dir / image.filename
    with open(file_path, "wb") as buffer:
        buffer.write(await image.read())

    image_url = f"/uploads/{image.filename}"

    try:
        model, tokenizer, max_length = get_model_bundle()
        caption = generate_caption(file_path, model, tokenizer, max_length - 1)
    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "caption": None,
                "image_name": image.filename,
                "image_url": image_url,
                "error": (
                    "Prediction could not be generated because the ViT encoder "
                    "dependency is not compatible with the current TensorFlow environment. "
                    f"Details: {exc}"
                ),
            },
        )

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "caption": caption,
            "image_name": image.filename,
            "image_url": image_url,
            "error": None,
        },
    )
