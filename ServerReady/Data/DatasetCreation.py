import pandas as pd
import numpy as np
import torch
import argparse
import os
from tqdm import tqdm


def main(local=False):
    """
    Create the dataset from the CSV file and save it as a .pt file.

    Args:
        local: If True, use relative paths. If False, use server paths.
    """

    # -----------------------------
    # Paths
    # -----------------------------
    if local:
        _dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(_dir, "..", "..", "RingkøbingData.csv")
        output_path = os.path.join(_dir, "..", "ModelTuning", "dataset.pt")
        print("Running in LOCAL mode (relative paths)")
    else:
        csv_file = "/ceph/project/SW6-Group18-Abvaerk/RingkøbingData.csv"
        output_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/dataset.pt"
        print("Running in SERVER mode (absolute paths)")

    # -----------------------------
    # Config
    # -----------------------------
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
    df['abvaerk'] = df['abvaerk'].interpolate(method='linear').bfill().ffill()

    forecast_features = ['toutdoor', 'temperature', 'relativeHumidity', 'windSpeed', 'precipitation', 'cloudCover']
    for col in forecast_features:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').bfill().ffill()

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
        encoder_slice = df.iloc[i:i+encoder_history][encoder_features].values.astype(np.float32)

        decoder_time_slice = df.iloc[i+encoder_history:i+encoder_history+forecast_length][decoder_time_features].values.astype(np.float32)

        decoder_forecast_slice = np.zeros((forecast_length, 5), dtype=np.float32)
        for j, cols in enumerate(forecast_cols_all):
            decoder_forecast_slice[:, j] = df.iloc[i+encoder_history:i+encoder_history+forecast_length][cols].values[:, np.arange(forecast_length)].diagonal()

        decoder_slice = np.concatenate([decoder_time_slice, decoder_forecast_slice], axis=1)

        target_slice = df.iloc[i+encoder_history:i+encoder_history+forecast_length]['abvaerk'].values.astype(np.float32)

        encoder_data.append(encoder_slice)
        decoder_data.append(decoder_slice)
        target_data.append(target_slice)

    encoder_data = np.stack(encoder_data)
    decoder_data = np.stack(decoder_data)
    target_data = np.stack(target_data)

    torch.save({
        'encoder': torch.from_numpy(encoder_data),
        'decoder': torch.from_numpy(decoder_data),
        'target': torch.from_numpy(target_data)
    }, output_path)

    print(f"Dataset saved to {output_path}")
    return output_path


# Allow standalone execution: python3 DatasetCreation.py --local
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Use local relative paths instead of server paths')
    args = parser.parse_args()
    main(local=args.local)
