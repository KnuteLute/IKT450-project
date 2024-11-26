import os
import json
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# Step 0: Set seeds for reproducibility
seed_value = 42

# 1. Set the PYTHONHASHSEED environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the Python built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set the NumPy pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the TensorFlow pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

# Optional: Limit GPU memory growth (helps with reproducibility and memory errors)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Clear any previous sessions.
K.clear_session()

import json
import numpy as np

# Load the sentiment and price movement data from JSON files
with open('average_daily_sentiment_by_stock.json', 'r') as f:
    sentiment_data = json.load(f)

with open('price_movement_by_stock.json', 'r') as f:
    price_movement_data = json.load(f)

    
def get_day_of_week(date_str):
    """
    Convert a date string (YYYY-MM-DD) into a day of the week as an integer.
    Monday=0, Sunday=6.
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj.weekday()

    
# Prepare data for all stocks
def prepare_data_all_stocks(sentiment_data, price_movement_data, sequence_length=10):
    """
    Prepare the input sequences and labels for all stocks.
    Includes day of the week as an input feature.
    
    Args:
        sentiment_data: Dictionary containing sentiment data for stocks.
        price_movement_data: Dictionary containing price movement data for stocks.
        sequence_length: Length of the input sequence for each stock.
        
    Returns:
        X_all: Numpy array of input sequences.
        y_all: Numpy array of labels.
    """
    X_all = []
    y_all = []

    for stock_name, data_by_date in price_movement_data.items():
        # Skip stocks with insufficient data
        if len(data_by_date) < sequence_length:
            continue

        # Sort dates for chronological order
        sorted_dates = sorted(data_by_date.keys())

        # Create sequences and labels
        for i in range(len(sorted_dates) - sequence_length):
            sequence_dates = sorted_dates[i:i + sequence_length]
            label_date = sorted_dates[i + sequence_length]

            # Ensure the label date exists in data
            if label_date not in data_by_date:
                continue

            sequence = []
            valid_sequence = True
            for j, current_date in enumerate(sequence_dates):
                # Get data for the current date
                if current_date not in data_by_date:
                    valid_sequence = False
                    break
                date_data = data_by_date[current_date]
                if len(date_data) < 3:
                    valid_sequence = False
                    break  # Incomplete data
                sentiment_bool = date_data[0]
                sentiment = 1 if sentiment_bool else 0  # Convert boolean to integer
                price_movement = date_data[1]
                volume = date_data[2]
                day_of_week = get_day_of_week(current_date)  # Convert date to day of week

                # Calculate sentiment change from the previous day
                if j > 0:
                    prev_date = sequence_dates[j - 1]
                    prev_date_data = data_by_date.get(prev_date, None)
                    if prev_date_data is None or len(prev_date_data) < 1:
                        prev_sentiment = sentiment  # Default to current sentiment
                    else:
                        prev_sentiment_bool = prev_date_data[0]
                        prev_sentiment = 1 if prev_sentiment_bool else 0
                    sentiment_change = sentiment - prev_sentiment
                else:
                    sentiment_change = 0  # No change for the first day

                # Add features to the sequence: [sentiment, sentiment_change, price_movement, volume, day_of_week]
                sequence.append([sentiment, sentiment_change, price_movement, volume, day_of_week])

            if not valid_sequence:
                continue  # Skip sequences with missing data

            # Append the sequence and its label
            label_data = data_by_date[label_date]
            if len(label_data) < 2:
                continue  # Skip if label data is incomplete
            label_price_movement = label_data[1]
            y_label = 1 if label_price_movement > 0 else 0  # Binary label: 1 for upward movement, 0 otherwise

            X_all.append(sequence)
            y_all.append(y_label)

    # Convert to NumPy arrays
    return np.array(X_all), np.array(y_all)
# Call the function to prepare data
sequence_length = 10
X, y = prepare_data_all_stocks(sentiment_data, price_movement_data, sequence_length=sequence_length)

# Print the shapes of X and y
print("X shape:", X.shape)  # Expected shape: (num_samples, sequence_length, num_features)
print("y shape:", y.shape)  # Expected shape: (num_samples,)



# Step 3: Shuffle the data while maintaining the correspondence between X and y
X, y = shuffle(X, y, random_state=seed_value)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed_value, shuffle=True
)

# Step 5: Standardize the features
# Reshape data to 2D for scaling
num_samples_train, timesteps, num_features = X_train.shape
num_samples_test = X_test.shape[0]

X_train_reshaped = X_train.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# Reshape back to 3D
X_train = X_train_scaled.reshape(num_samples_train, timesteps, num_features)
X_test = X_test_scaled.reshape(num_samples_test, timesteps, num_features)

# Step 6: Build the model 
# Switch SimpleRNN with LSTM and GRU for other model. 
# Add Bidirectonal() around the SimpleRNN to add bidireconal to the model.
model = Sequential()
model.add(SimpleRNN(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(SimpleRNN(64, activation='tanh', return_sequences=True))
model.add(SimpleRNN(32, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 7: Train the model
epochs = 50
batch_size = 16
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Step 8: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

# Step 9: Collect metrics into a dictionary
results = {
    "train_loss": history.history['loss'],
    "train_accuracy": history.history['accuracy'],
    "val_loss": history.history['val_loss'],
    "val_accuracy": history.history['val_accuracy'],
    "test_loss": loss,
    "test_accuracy": accuracy
}

# Step 10: Save the results into a JSON file, appending if the file already exists
file_name = 'model_metrics.json'

# Check if the file exists
if os.path.exists(file_name):
    # Load existing data
    with open(file_name, 'r') as f:
        all_results = json.load(f)
else:
    # Initialize a new dictionary if the file doesn't exist
    all_results = {}

# Generate a unique key for the current run (e.g., LSTM_1, GRU_2)
model_type = "LSTM"  # Change this depending on your model type
run_id = len(all_results) + 1
current_run_key = f"{model_type}_run_{run_id}"

# Add the new results to the dictionary
all_results[current_run_key] = {
    "train_loss": history.history['loss'],
    "train_accuracy": history.history['accuracy'],
    "val_loss": history.history['val_loss'],
    "val_accuracy": history.history['val_accuracy'],
    "test_loss": loss,
    "test_accuracy": accuracy
}

# Write the updated results back to the file
with open(file_name, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"Results saved under key '{current_run_key}' in {file_name}")
