import streamlit as st
import types
import random
from transformers import FlaubertTokenizer, TFFlaubertForSequenceClassification
import tensorflow as tf
import numpy as np
import pandas as pd

# Fonction pour charger les phrases du fichier CSV
def load_phrases():
    try:
        df = pd.read_csv('unlabelled_test_data.csv')
        return df['sentence'].tolist()
    except FileNotFoundError:
        st.error("Le fichier 'unlabelled_test_data.csv' n'a pas été trouvé.")
        return []

phrases = load_phrases()

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

def predict_level(phrase, tokenizer, model):
    # Préparation de la phrase pour le modèle
    input_ids, attention_masks = encode_text(phrase, tokenizer)
    
    # Prédiction du niveau de difficulté
    predictions = model.predict([input_ids, attention_masks])
    difficulty_level = np.argmax(predictions.logits, axis=1)[0]

    # Mapping du niveau prédit à son label correspondant
    levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    return levels[difficulty_level]

# Load FlauBERT tokenizer
tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')

# Initialisation des variables de session pour le jeu
if 'score' not in st.session_state:
    st.session_state['score'] = 0
if 'phrase_count' not in st.session_state:
    st.session_state['phrase_count'] = 0
if 'current_phrase' not in st.session_state:
    st.session_state['current_phrase'] = None
if 'game_history' not in st.session_state:
    st.session_state['game_history'] = []
if 'user_guess' not in st.session_state:
    st.session_state['user_guess'] = None
    
# Load the model
model = load_model()

# Streamlit interface
st.title('French Text Difficulty Application')

# Menu de sélection dans la barre latérale
option = st.sidebar.selectbox(
    'Choisissez une option',
    ('Prédiction de Phrase', 'Jeu de Prédiction de Niveau')
)

# Logique conditionnelle en fonction de l'option sélectionnée
if option == 'Prédiction de Phrase':
    st.subheader("Prédiction de Niveau de Difficulté d'une Phrase")
    user_input = st.text_area("Entrez une phrase en français", "")
    if st.button('Prédire le Niveau'):
        with st.spinner('Analyse en cours...'):
            input_ids, attention_masks = encode_text(user_input, tokenizer)
            predictions = model.predict([input_ids, attention_masks])
            difficulty_level = np.argmax(predictions.logits, axis=1)[0]
            levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
            predicted_level = levels[difficulty_level]
            st.success(f"Le niveau de difficulté prédit est : {predicted_level}")

elif option == 'Jeu de Prédiction de Niveau':
    st.subheader("Jeu de Prédiction de Niveau de Langue")

    if st.session_state['phrase_count'] < 10:
        if st.session_state['current_phrase'] is None:
            st.session_state['current_phrase'] = random.choice(phrases)
            st.session_state['user_guess'] = None  # Réinitialiser la supposition de l'utilisateur

        st.write(st.session_state['current_phrase'])
        st.session_state['user_guess'] = st.selectbox("Quel est le niveau de cette phrase ?", ["A1", "A2", "B1", "B2", "C1", "C2"], key=st.session_state['phrase_count'])

        if st.button("Valider"):
            predicted_level = predict_level(st.session_state['current_phrase'], tokenizer, model)
            if st.session_state['user_guess'] == predicted_level:
                st.session_state['score'] += 1
                correct = True
            else:
                correct = False

            # Enregistrer l'historique du jeu
            st.session_state['game_history'].append((st.session_state['current_phrase'], st.session_state['user_guess'], predicted_level, correct))
            st.session_state['current_phrase'] = None
            st.session_state['phrase_count'] += 1

    else:
        st.subheader(f"Votre score : {st.session_state['score']} / 10")
        # Afficher l'historique du jeu avec les corrections
        for phrase, user_guess, predicted_level, correct in st.session_state['game_history']:
            if correct:
                st.markdown(f"**{phrase}** - Correct !")
            else:
                st.markdown(f"**{phrase}** - Votre réponse : {user_guess}, Niveau prédit : {predicted_level}")

        if st.button("Recommencer le Jeu"):
            st.session_state['score'] = 0
            st.session_state['phrase_count'] = 0
            st.session_state['current_phrase'] = None
            st.session_state['game_history'] = []
            st.session_state['user_guess'] = None
