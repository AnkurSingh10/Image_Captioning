import numpy as np
import pandas as pd
from layers.Custom_layer_model import Transformer_decoder, Expand_Dimension, Masked_Loss, PositionalEmbedding
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow import keras
from keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from models.feature_engineering import load_caption, clean_caption, tokenize_captions, split_dataset, extract_features
from layers.data_gen import CustomDataGenerator


#load caption and clean it
caption = load_caption("flickr30k/captions.txt")
caption['comment'] = caption['comment'].apply(clean_caption)

#tokenize captions
tokenizer, vocab_size, max_length = tokenize_captions(caption)

#split dataset into train and validation
train_df, val_df, test_df = split_dataset(caption)

test_df.to_csv("data/test.csv", index=False)

#feature extraction from images using VIT model
features = extract_features(caption, "flickr30k/flickr30k_images")
for k in features:
    features[k] = np.squeeze(features[k])

# train and validation data generators
train_generator = CustomDataGenerator(train_df, features, tokenizer, max_length, batch_size=32, shuffle=True)
val_generator   = CustomDataGenerator(val_df, features, tokenizer, max_length, batch_size=32, shuffle=False)



def define_model(
    vocab_size,
    max_length,
    embed_dimension=256,
    ff_dimension=512,
    num_heads=8,
    num_layers=4,
    
):
    img_input = Input(shape=(768,), name="image_features")
    cap_input = Input(shape=(max_length-1,), name="caption_input")
    
    img_emb = Expand_Dimension(name="image_context")(img_input)
    decoder = Transformer_decoder(embed_dimension=embed_dimension, ff_dimension=ff_dimension, num_heads=num_heads,
                                 vocab_size=vocab_size, max_len=max_length-1,
                                 num_layers=num_layers, rate=0.1)
    
    outputs = decoder(cap_input, img_emb) 
    
    model = Model(inputs=[img_input, cap_input], outputs=outputs)
    model.summary()
    return model
  
#define model
model = define_model(
    vocab_size=vocab_size,
    max_length=max_length,
)

#compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss=Masked_Loss)

# plot model architecture
plot_model(
    model,
    to_file="caption_model.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True
)

check_point = ModelCheckpoint("caption_model.keras", monitor="val_loss",
                             save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=5,
                          restore_best_weights=True, verbose=1)
reducelr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, verbose=1)
callbacks = [
    check_point,
    reducelr,
    early_stopping
]

model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

#save model
model.save("caption_model.h5")

#load model
model = load_model("caption_model.h5",
                   custom_objects={"Transformer_decoder": Transformer_decoder,
                                   "PositionalEmbedding": PositionalEmbedding,
                                   "Masked_Loss": Masked_Loss})