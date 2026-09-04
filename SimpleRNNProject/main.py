import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
model  = load_model('model_final.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

sentiment_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}

st.title('Twitter Tweets Sentiment Analysis ')
tweet = st.text_area("Enter the tweet:")

if st.button('Predict Sentiment') and tweet.strip():
    sequence = tokenizer.texts_to_sequences([tweet])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)[0]
    sentiment = sentiment_map.get(predicted_class, 'Unknown')
    st.write(f"Predicted Sentiment: {sentiment}")

