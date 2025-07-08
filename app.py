import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved files
model = load_model("nextword_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_sequence_len.pkl", "rb") as f:
    max_sequence_len = pickle.load(f)

# Prediction function
def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]
    predicted_index = np.argmax(predicted_probs)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return "No prediction"

# Streamlit UI
st.title("🔮 Next Word Predictor")
user_input = st.text_input("Enter a sentence:")

if user_input:
    next_word = predict_next_word(user_input)
    st.markdown(f"### ➡️ Predicted Next Word: **{next_word}**")
