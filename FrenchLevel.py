!pip install streamlit
!pip install transformers
!pip install tensorflow
!pip install sacremoses
!pip install sentencepiece


import streamlit as st
import types
from transformers import FlaubertTokenizer, TFFlaubertForSequenceClassification
import tensorflow as tf
import numpy as np

# Custom hash function to bypass hashing of the load_model function
def bypass_hashing(func):
    return 0

# Function to load the FlauBERT model
@st.cache(allow_output_mutation=True, hash_funcs={types.FunctionType: bypass_hashing})
def load_model():
    model = TFFlaubertForSequenceClassification.from_pretrained('flaubert/flaubert_base_cased', num_labels=6, from_pt=True)
    return model

# Function to encode text for FlauBERT
def encode_text(text, tokenizer, max_length=128):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='tf',
    )
    return encoded_dict['input_ids'], encoded_dict['attention_mask']

# Load FlauBERT tokenizer
tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')

# Load the model
model = load_model()

# Streamlit interface
st.title('French Text Difficulty Predictor')
user_input = st.text_area("Enter a sentence in French", "")

if st.button('Predict Difficulty'):
    input_ids, attention_masks = encode_text(user_input, tokenizer)
    predictions = model.predict([input_ids, attention_masks])
    difficulty_level = np.argmax(predictions.logits, axis=1)[0]

    # Mapping the prediction to difficulty level
    levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    predicted_level = levels[difficulty_level]

    st.write(f"The predicted difficulty level is: {predicted_level}")
