import streamlit as st
from textblob import TextBlob
import pandas as pd
import pickle

# Function to make predictions using TextBlob
def predict_sentiment(text):
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity

    if sentiment > 0:
        return 'Positive'
    elif sentiment == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Function to generate sample data from the dataset
def get_sample_data(file_path, num_samples=5):
    try:
        data = pd.read_csv(file_path)
        data = data.dropna(subset=['Title', 'Description', 'Sentiment'])
        data['Text'] = data['Title'] + " " + data['Description']
        
        # Filter data to include only rows containing "Coca-Cola"
        data = data[data['Text'].str.contains("Coca-Cola", case=False, na=False)]
        
        sample_data = {}
        sentiments = ['Negative', 'Neutral', 'Positive']
        
        for sentiment in sentiments:
            # Filter for each sentiment and sample a few examples
            if sentiment in data['Sentiment'].values:
                samples = data[data['Sentiment'] == sentiment]['Text'].sample(n=num_samples, random_state=42).tolist()
                sample_data[sentiment] = samples
            else:
                sample_data[sentiment] = [f"No samples available for {sentiment}."]
        
        return sample_data
    except Exception as e:
        st.error(f"Error loading or processing the dataset: {e}")
        return {}

# Path to the dataset
dataset_path = 'synthetic_coca_cola_sentiment_analysis.csv'

# Streamlit app
st.title('Coca-Cola Sentiment Analysis')

# Dropdown menu
option = st.selectbox(
    'Choose an option:',
    ['Sentiment Prediction', 'Generate Sample Data']
)

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

if option == 'Sentiment Prediction':
    st.write("Enter some text about Coca-Cola to analyze its sentiment.")
    st.session_state.user_input = st.text_area("Text Input", value=st.session_state.user_input)
    
    if st.button("Analyze Sentiment"):
        if st.session_state.user_input:
            sentiment = predict_sentiment(st.session_state.user_input)
            st.write(f"Predicted Sentiment: {sentiment}")
        else:
            st.write("Please enter some text for analysis.")

elif option == 'Generate Sample Data':
    if st.button("Generate Sample Data"):
        sample_data = get_sample_data(dataset_path)
        st.write("Sample Sentiments:")
        for sentiment, texts in sample_data.items():
            st.write(f"**{sentiment}**:")
            for text in texts:
                st.write(f"- {text}")
