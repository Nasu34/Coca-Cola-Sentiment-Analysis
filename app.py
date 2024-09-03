import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained SVM model and TF-IDF vectorizer
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to make predictions
def predict_sentiment(text):
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    
    # Map numerical prediction to sentiment labels
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map.get(prediction[0], 'Unknown')

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

# Function to perform EDA
def perform_eda(file_path):
    try:
        data = pd.read_csv(file_path)
        data = data.dropna(subset=['Sentiment'])
        
        # Sentiment distribution pie chart
        sentiment_counts = data['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)
        
        # Word clouds for each sentiment
        sentiments = ['Negative', 'Neutral', 'Positive']
        for sentiment in sentiments:
            subset = data[data['Sentiment'] == sentiment]
            text = " ".join(subset['Title'].astype(str) + " " + subset['Description'].astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            st.write(f"**Word Cloud for {sentiment} Sentiment**")
            st.image(wordcloud.to_image())
    except Exception as e:
        st.error(f"Error performing EDA: {e}")

# Path to the dataset
dataset_path = 'synthetic_coca_cola_sentiment_analysis.csv'

# Streamlit app
st.title('Coca-Cola Sentiment Analysis')

# Dropdown menu
option = st.selectbox(
    'Choose an option:',
    ['Sentiment Prediction', 'Generate Sample Data', 'Exploratory Data Analysis']
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

elif option == 'Exploratory Data Analysis':
    perform_eda(dataset_path)
