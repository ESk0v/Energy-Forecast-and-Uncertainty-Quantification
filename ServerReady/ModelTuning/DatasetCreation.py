import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
csv_file = "RingkøbingData.csv"  # your CSV path
encoder_history = 168  # 1 week of past data
forecast_length = 168  # 1 week forecast

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(csv_file, parse_dates=["dateTime"])
print(f"Loaded {len(df)} rows.")

# -----------------------------
# Interpolate missing values (instead of filling with 0)
# -----------------------------
# Interpolate abvaerk
df['abvaerk'] = df['abvaerk'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# Interpolate other forecast columns
forecast_features = ['toutdoor', 'temperature', 'relativeHumidity', 'windSpeed', 'precipitation', 'cloudCover']
for col in forecast_features:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# -----------------------------
# Prepare forecast column lists (week ahead)
# -----------------------------
temperature_cols = [f"temperature_{i}" for i in range(forecast_length)]
humidity_cols = [f"relativeHumidity_{i}" for i in range(forecast_length)]
wind_cols = [f"windSpeed_{i}" for i in range(forecast_length)]
precip_cols = [f"precipitation_{i}" for i in range(forecast_length)]
cloud_cols = [f"cloudCover_{i}" for i in range(forecast_length)]

forecast_cols_all = [temperature_cols, humidity_cols, wind_cols, precip_cols, cloud_cols]

# -----------------------------
# Prepare time features
# -----------------------------
def get_time_features(df):
    hour = df['dateTime'].dt.hour
    weekday = df['dateTime'].dt.weekday
    month = df['dateTime'].dt.month

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    return df

df = get_time_features(df)

# -----------------------------
# Build dataset
# -----------------------------
encoder_features = ['abvaerk', 'toutdoor', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']
decoder_time_features = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']

encoder_data = []
decoder_data = []
target_data = []

for i in tqdm(range(len(df) - encoder_history - forecast_length)):
    # Encoder input
    encoder_slice = df.iloc[i:i+encoder_history][encoder_features].values.astype(np.float32)

    # Decoder time features
    decoder_time_slice = df.iloc[i+encoder_history:i+encoder_history+forecast_length][decoder_time_features].values.astype(np.float32)

    # Decoder forecast: take **hour-aligned value** for each forecast type
    decoder_forecast_slice = np.zeros((forecast_length, 5), dtype=np.float32)
    for j, cols in enumerate(forecast_cols_all):
        # pick the correct hour index for each column
        decoder_forecast_slice[:, j] = df.iloc[i+encoder_history:i+encoder_history+forecast_length][cols].values[:, np.arange(forecast_length)].diagonal()

    # Concatenate time + forecast
    decoder_slice = np.concatenate([decoder_time_slice, decoder_forecast_slice], axis=1)

    # Target
    target_slice = df.iloc[i+encoder_history:i+encoder_history+forecast_length]['abvaerk'].values.astype(np.float32)

    encoder_data.append(encoder_slice)
    decoder_data.append(decoder_slice)
    target_data.append(target_slice)

# Stack arrays
encoder_data = np.stack(encoder_data)
decoder_data = np.stack(decoder_data)
target_data = np.stack(target_data)

# Save dataset
torch.save({
    'encoder': torch.from_numpy(encoder_data),
    'decoder': torch.from_numpy(decoder_data),
    'target': torch.from_numpy(target_data)
}, "dataset.pt")

print("Dataset saved as dataset.pt")
