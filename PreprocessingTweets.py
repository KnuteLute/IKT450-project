from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import os
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score from a tweet
def get_sentiment(tweet_text):
    sentiment_dict = analyzer.polarity_scores(tweet_text)  # Analyze the sentiment
    return sentiment_dict['compound']  # Return the compound sentiment score

# Function to extract the date in "YYYY-MM-DD" format from 'created_at'
def extract_date(created_at_str):
    date_obj = datetime.strptime(created_at_str, '%a %b %d %H:%M:%S %z %Y')  # Parse the created_at string
    return date_obj.strftime('%Y-%m-%d')  # Return date as "YYYY-MM-DD"

# Directory containing the stock folders
base_dir = 'stocknet-dataset/tweet/raw/'

stock_sentiment_data = {}

# Loop through all folders (stocks) in the base directory
for stock_folder in tqdm(os.listdir(base_dir)):
    stock_dir = os.path.join(base_dir, stock_folder)
    
    if os.path.isdir(stock_dir) and not stock_folder.startswith('.'):  # Skip hidden directories
        print(f"Processing stock: {stock_folder}")
        
        # Dictionary to store the sum of compound scores and count for each day for the current stock
        daily_sentiment = defaultdict(lambda: {'total_score': 0, 'count': 0, 'tweets': []})
        
        # Loop through all files in the stock folder
        for filename in os.listdir(stock_dir):
            file_path = os.path.join(stock_dir, filename)
            
            # Skip directories or hidden files
            if os.path.isdir(file_path) or filename.startswith('.'):
                continue
            
            # Open the file and process each line (each line is a JSON object)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Parse each line as a JSON object
                        tweet_data = json.loads(line.strip())
                        
                        # Check if 'text' and 'created_at' are in tweet_data
                        if 'text' in tweet_data and 'created_at' in tweet_data:
                            tweet_text = tweet_data['text']
                            compound_score = get_sentiment(tweet_text)
                            tweet_date = extract_date(tweet_data['created_at'])  # Get the date in "YYYY-MM-DD"
                            
                            # Accumulate the compound score, increment count, and save tweet text
                            daily_sentiment[tweet_date]['total_score'] += compound_score
                            daily_sentiment[tweet_date]['count'] += 1
                            daily_sentiment[tweet_date]['tweets'].append({
                                'text': tweet_text,
                                'score': compound_score
                            })

                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {filename}, skipping line")
                        continue

        # Store results for the current stock, including all sentiment scores
        stock_sentiment_data[stock_folder] = {
            date: {
                'average_score': round(data['total_score'] / data['count'], 4),  # Average score rounded to 4 decimal places
                'tweets': data['tweets']  # List of tweets with their sentiment scores
            }
            for date, data in daily_sentiment.items()
        }

# Save the result to a JSON file
output_file = 'daily_sentiment_by_stock.json'
with open(output_file, 'w', encoding='utf-8') as json_out:
    json.dump(stock_sentiment_data, json_out, indent=4)

print(f"Detailed sentiment by day has been saved to {output_file}")
