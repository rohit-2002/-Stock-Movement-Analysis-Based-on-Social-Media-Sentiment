# Stock Movement Analysis Based on Social Media Sentiment

This project collects tweets related to the stock market, performs sentiment analysis, and trains a sentiment prediction model to classify tweets as positive or negative. The goal is to analyze social media sentiment and explore its potential for predicting stock market movements.

## Features

- Collects recent tweets from Twitter using the Tweepy API.
- Cleans tweet text by removing URLs, mentions, and hashtags.
- Performs sentiment analysis on tweets using TextBlob.
- Trains a Random Forest Classifier to predict positive or negative sentiment based on the sentiment scores.

## Prerequisites

- Python 3.6 or higher
- Git (to clone the repository)
- An active Twitter Developer account (for API credentials)

## Setup Instructions

 1. Clone the Repository

    Clone this repository to your local machine using Git.

    ```bash
    git clone https://github.com/rohit-2002/Stock-Movement-Analysis-Based-on-Social-Media-Sentiment.git
    cd Stock-Movement-Analysis-Based-on-Social-Media-Sentiment
    ```
2. Install Dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   pip install -r requirements.txt

3. Set Up Twitter API Credentials
   - To interact with Twitter's API, you'll need to set up your own API credentials:
   - Go to Twitter Developer Portal.
   - Create a new application and get your API credentials (API Key, API Secret, Access Token, Access Token Secret, Bearer Token).
   - Create a .env file in the project root and add the following environment variables:
     
     ```bash
     TWITTER_API_KEY=your_api_key
     TWITTER_API_SECRET=your_api_secret
     ACCESS_TOKEN=your_access_token
     ACCESS_TOKEN_SECRET=your_access_token_secret
     BEARER_TOKEN=your_bearer_token
     
4. Run the Pipeline
   - Once everything is set up, you can run the script to execute the entire process. This will:
   - Collect tweets based on a search term (e.g., "stock market").
   - Clean and preprocess the tweet data.
   - Perform sentiment analysis on the tweets.
   - Train a sentiment prediction model.
     
     ```bash
     python run_all.py
     ```
     
   - This will generate several CSV files in the data/ directory, including:
   - raw_tweets.csv — The collected tweets.
   - processed_tweets.csv — The cleaned tweet data.
   - sentiment_tweets.csv — The tweets with sentiment scores.
   - sentiment_model.pkl — The trained Random Forest model.


