import os
import json
from tqdm import tqdm

# Directory containing the preprocessed price data
base_dir = 'stocknet-dataset/price/preprocessed/'

# Dictionary to store the price movement results for all stocks
price_movement_data = {}

# Loop through all stock files in the base directory
for stock_file in tqdm(os.listdir(base_dir)):
    file_path = os.path.join(base_dir, stock_file)
    if os.path.isdir(file_path):
        continue

    stock_name, _ = os.path.splitext(stock_file)  # Get the stock name (file without extension)

    # Dictionary to store the price movement (True/False) for each date
    price_movement = {}

    # Read the preprocessed price data from the .txt file
    with open(file_path, 'r', encoding='utf-8') as f:
        previous_close_price = None  # To track the previous day's close price
        
        # Loop through each line in the file (assuming tab-separated)
        for line in f:
            # Split the line into components (date, movement percent, open, high, low, close, volume)
            line_data = line.strip().split()
            if len(line_data) < 7:
                continue  # Skip lines that don't have enough data

            # Extract relevant fields
            date, movement_percent, open_price, high_price, low_price, close_price, volume = line_data

            # Convert the close price to a float for comparison
            close_price = float(close_price)

            # Determine if the price has risen compared to the previous day
            if previous_close_price is not None:
                price_differace = close_price - previous_close_price
                price_has_risen = close_price > previous_close_price
                price_movement[date] = price_has_risen, price_differace, float(volume), open_price, high_price, low_price, close_price, # Store True if risen, False if not

            # Update the previous close price
            previous_close_price = close_price

    # Store the results for the current stock
    price_movement_data[stock_name] = price_movement

# Save the result to a JSON file
output_file = 'price_movement_by_stock.json'
with open(output_file, 'w', encoding='utf-8') as json_out:
    json.dump(price_movement_data, json_out, indent=4)

print(f"Price movement data has been saved to {output_file}")
