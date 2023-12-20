import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from transformers import FlaubertTokenizer, TFFlaubertModel, TFFlaubertForSequenceClassification
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Charger les données
train_data = pd.read_csv('training_data.csv')

# Dupliquer les données une première fois
train_data_duplicated_once = pd.concat([train_data, train_data])

# Dupliquer les données une deuxième fois pour obtenir une multiplication par 4
train_data_duplicated_twice = pd.concat([train_data_duplicated_once, train_data_duplicated_once])

train_data = train_data_duplicated_twice

X = train_data['sentence']
y = train_data['difficulty']

# Encodage des labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialiser le tokenizer FlauBERT
tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')

# Ppar les données pour FlauBERT
def encode_for_flaubert(sentences, max_length=128):
    input_ids = []
    attention_masks = []

    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            turn_attention_mask=True,
            turn_tensors='tf',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = tf.concat(input_ids, 0)
    attention_masks = tf.concat(attention_masks, 0)

    return input_ids, attention_masks

train_input_ids, train_attention_masks = encode_for_flaubert(X_train, max_length=128)
test_input_ids, test_attention_masks = encode_for_flaubert(X_test, max_length=128)


# Charger le modèle FlauBERT p-entraîné pour la classification de séquence
model = TFFlaubertForSequenceClassification.from_pretrained('flaubert/flaubert_base_cased', num_labels=len(label_encoder.classes_), from_pt=True)

# Compiler le modèle
optimizer = Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Entraîner le modèle
model.fit(
    [train_input_ids, train_attention_masks],
    y_train,
    epochs=1,
    batch_size=16,
    validation_data=([test_input_ids, test_attention_masks], y_test)
)




# Stamlit interface
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
