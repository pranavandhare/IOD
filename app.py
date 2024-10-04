import streamlit as st
import pandas as pd
import joblib
from utils import preprocessor

def run():
    model = joblib.load('model.joblib')

    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text.")
    st.text("")
    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")
    
    predicted_sentiment = ""
    if st.button("Predict"):
         # Preprocess the input text
        processed_input = pd.Series([preprocessor().transform(pd.Series([userinput]))[0]])
        
        # Get the prediction from the model
        predicted_sentiment = model.predict(processed_input)[0]
        
        if predicted_sentiment == 1:
            output = 'positive 👍'
        else:
            output = 'negative 👎'
        sentiment=f'Predicted sentiment of "{userinput}" is {output}.'
        st.success(sentiment)

if __name__ == "__main__":
    run()