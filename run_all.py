import time
import os
import tweepy
import pandas as pd
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Twitter API credentials
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Initialize Tweepy client
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Function to collect tweets from Twitter API
def collect_tweets(search_term, max_tweets=100):
    collected_tweets = []
    try:
        for tweet_batch in tweepy.Paginator(client.search_recent_tweets, query=search_term, max_results=10, limit=max_tweets // 10):
            for tweet in tweet_batch.data:
                collected_tweets.append(tweet.text)
                print(f"Collected {len(collected_tweets)} tweets...")

            # Stop if maximum tweets are collected
            if len(collected_tweets) >= max_tweets:
                break

    except tweepy.errors.TooManyRequests as error:
        reset_timestamp = error.response.headers.get('x-rate-limit-reset')
        if reset_timestamp:
            sleep_duration = int(reset_timestamp) - int(time.time()) + 5  # Adding buffer
            print(f"Rate limit reached. Sleeping for {sleep_duration} seconds.")
            time.sleep(sleep_duration)
        else:
            print("Rate limit exceeded, retrying...")
            time.sleep(60)  # Sleep for 1 minute

    # Save tweets to CSV
    tweet_df = pd.DataFrame(collected_tweets, columns=['text'])
    tweet_df.to_csv("data/raw_tweets.csv", index=False)
    print("Tweets saved as raw_tweets.csv")
    return "data/raw_tweets.csv"

# Function to clean tweet text by removing unnecessary parts
def clean_tweet_content(tweet_text):
    tweet_text = re.sub(r"http\S+", "", tweet_text)  # Remove URLs
    tweet_text = re.sub(r"@\w+", "", tweet_text)    # Remove mentions
    tweet_text = re.sub(r"#", "", tweet_text)       # Remove hashtags
    tweet_text = re.sub(r"\s+", " ", tweet_text)    # Clean extra spaces
    return tweet_text.strip()

# Process raw data by cleaning tweet content
def process_raw_data(file_path):
    data = pd.read_csv(file_path)
    data["cleaned_text"] = data["text"].apply(clean_tweet_content)
    data.to_csv("data/processed_tweets.csv", index=False)
    print("Cleaned tweet data saved as processed_tweets.csv")
    return "data/processed_tweets.csv"

# Function to perform sentiment analysis on tweet text
def get_sentiment(tweet_text):
    sentiment = TextBlob(tweet_text)
    return sentiment.sentiment.polarity

# Add sentiment scores to the cleaned data
def add_sentiment_scores(file_path):
    data = pd.read_csv(file_path)
    data["sentiment"] = data["cleaned_text"].apply(get_sentiment)
    data.to_csv("data/sentiment_tweets.csv", index=False)
    print("Sentiment analysis added and saved as sentiment_tweets.csv")
    return "data/sentiment_tweets.csv"

# Function to train a sentiment prediction model
def train_sentiment_model(file_path):
    data = pd.read_csv(file_path)
    X = data['sentiment'].values.reshape(-1, 1)  # Feature: sentiment polarity
    y = (data['sentiment'] > 0).astype(int)      # Target: Positive (1) or Negative (0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Save the trained model
    import joblib
    joblib.dump(model, "sentiment_model.pkl")
    print("Model saved as sentiment_model.pkl")

# Main function to execute all tasks sequentially
def execute_pipeline():
    print("Starting the process...")
    
    # Step 1: Collect tweets based on a search term
    collect_tweets("stock market", 100)
    
    # Step 2: Process the collected data (cleaning and formatting)
    process_raw_data("data/raw_tweets.csv")
    
    # Step 3: Add sentiment analysis to the cleaned data
    add_sentiment_scores("data/processed_tweets.csv")
    
    # Step 4: Train a sentiment prediction model
    train_sentiment_model("data/sentiment_tweets.csv")

# Run the main pipeline
if __name__ == "__main__":
    execute_pipeline()
