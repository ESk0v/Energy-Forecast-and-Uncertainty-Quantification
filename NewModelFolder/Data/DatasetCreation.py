import pandas as pd
import numpy as np
import torch
import argparse
from tqdm import tqdm
import logging


def main(local = False, filePaths = None, logger=None):
    csv_file = filePaths[0]
    output_path = filePaths[1]

    encoder_history = 168  # 1 week of past data
    forecast_length = 168  # 1 week forecast

    df = pd.read_csv(csv_file, parse_dates=["dateTime"])
    logger.info(f"Loaded dataset")

    # -----------------------------
    # Interpolate missing values (instead of filling with 0)
    # -----------------------------
    df['abvaerk'] = df['abvaerk'].interpolate(method='linear').bfill().ffill()

    forecast_features = ['toutdoor', 'temperature', 'relativeHumidity', 'windSpeed', 'precipitation', 'cloudCover']
    for col in forecast_features:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').bfill().ffill()
    
    # -----------------------------
    # Standardize demand (abvaerk)
    # -----------------------------
    demand_mean = df['abvaerk'].mean()
    demand_std  = df['abvaerk'].std()
    df['abvaerk'] = (df['abvaerk'] - demand_mean) / demand_std

    # -----------------------------
    # Normalise encoder: outdoor temperature (z-score)
    # -----------------------------
    toutdoor_mean = df['toutdoor'].mean()
    toutdoor_std  = df['toutdoor'].std()
    df['toutdoor'] = (df['toutdoor'] - toutdoor_mean) / toutdoor_std

    # -----------------------------
    # Normalise decoder forecast features
    # temperature  → z-score            (symmetric, contains negatives)
    # humidity     → z-score            (tight, symmetric)
    # cloud cover  → z-score            (bimodal but bounded)
    # wind speed   → log1p then z-score (right-skewed, occasional storm outliers)
    # precipitation→ log1p then z-score (82 % zeros, heavy right tail)
    # -----------------------------

    # Collect the raw forecast columns into flat series for stat computation.
    # We use only the _0 column family because all forecsast steps share the
    # same physical variable and should use consistent stats.

    def _flat(cols):
        """Stack all forecast-horizon columns into one 1-D array for fitting."""
        return df[cols].values.flatten()

    temperature_cols = [f"temperature_{i}"      for i in range(forecast_length)]
    humidity_cols    = [f"relativeHumidity_{i}"  for i in range(forecast_length)]
    wind_cols        = [f"windSpeed_{i}"         for i in range(forecast_length)]
    precip_cols      = [f"precipitation_{i}"     for i in range(forecast_length)]
    cloud_cols       = [f"cloudCover_{i}"        for i in range(forecast_length)]

    # -- temperature (z-score) -------------------------------------------
    temp_mean = float(np.mean(_flat(temperature_cols)))
    temp_std  = float(np.std (_flat(temperature_cols)))
    df[temperature_cols] = (df[temperature_cols] - temp_mean) / temp_std

    # -- humidity (z-score) -----------------------------------------------
    hum_mean = float(np.mean(_flat(humidity_cols)))
    hum_std  = float(np.std (_flat(humidity_cols)))
    df[humidity_cols] = (df[humidity_cols] - hum_mean) / hum_std

    # -- cloud cover (z-score) --------------------------------------------
    cloud_mean = float(np.mean(_flat(cloud_cols)))
    cloud_std  = float(np.std (_flat(cloud_cols)))
    df[cloud_cols] = (df[cloud_cols] - cloud_mean) / cloud_std

    # -- wind speed (log1p then z-score) ----------------------------------
    df[wind_cols] = np.log1p(df[wind_cols])
    wind_log_mean = float(np.mean(_flat(wind_cols)))
    wind_log_std  = float(np.std (_flat(wind_cols)))
    df[wind_cols] = (df[wind_cols] - wind_log_mean) / wind_log_std

    # -- precipitation (log1p then z-score) --------------------------------
    df[precip_cols] = np.log1p(df[precip_cols])
    precip_log_mean = float(np.mean(_flat(precip_cols)))
    precip_log_std  = float(np.std (_flat(precip_cols)))
    df[precip_cols] = (df[precip_cols] - precip_log_mean) / precip_log_std

    # Bundle all normalisation stats so downstream code can invert transforms
    norm_stats = {
        'demand_mean':      demand_mean,
        'demand_std':       demand_std,
        'toutdoor_mean':    toutdoor_mean,
        'toutdoor_std':     toutdoor_std,
        'temp_mean':        temp_mean,
        'temp_std':         temp_std,
        'hum_mean':         hum_mean,
        'hum_std':          hum_std,
        'cloud_mean':       cloud_mean,
        'cloud_std':        cloud_std,
        'wind_log_mean':    wind_log_mean,
        'wind_log_std':     wind_log_std,
        'precip_log_mean':  precip_log_mean,
        'precip_log_std':   precip_log_std,
    }

    logger.info(
        f"Normalisation stats — demand: mean={demand_mean:.4f}, std={demand_std:.4f} | "
        f"toutdoor: mean={toutdoor_mean:.4f}, std={toutdoor_std:.4f} | "
        f"temp: mean={temp_mean:.4f}, std={temp_std:.4f} | "
        f"hum: mean={hum_mean:.4f}, std={hum_std:.4f} | "
        f"cloud: mean={cloud_mean:.4f}, std={cloud_std:.4f} | "
        f"wind_log: mean={wind_log_mean:.4f}, std={wind_log_std:.4f} | "
        f"precip_log: mean={precip_log_mean:.4f}, std={precip_log_std:.4f}"
    )

    # -----------------------------
    # Prepare forecast column lists (week ahead)
    # NOTE: columns are already normalised above — no further scaling needed
    # -----------------------------
    forecast_cols_all = [temperature_cols, humidity_cols, wind_cols, precip_cols, cloud_cols]

    # -----------------------------
    # Prepare time features (fixed to avoid fragmentation warning)
    # -----------------------------
    def get_time_features(df):
        hour = df['dateTime'].dt.hour
        weekday = df['dateTime'].dt.weekday
        month = df['dateTime'].dt.month

        # Create all time features at once using pd.concat to avoid fragmentation
        time_features = pd.DataFrame({
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'weekday_sin': np.sin(2 * np.pi * weekday / 7),
            'weekday_cos': np.cos(2 * np.pi * weekday / 7),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12)
        }, index=df.index)
        
        # Concatenate once instead of adding columns one by one
        df = pd.concat([df, time_features], axis=1)
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

    logger.info("Building encoder/decoder/target tensors...")
    
    for i in tqdm(range(len(df) - encoder_history - forecast_length), disable=True):
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
        'target':  torch.from_numpy(target_data),
        **norm_stats,   # demand_mean/std, toutdoor, temp, hum, cloud, wind_log, precip_log
    }, output_path)

    return output_path