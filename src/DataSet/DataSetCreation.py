import json
import pathlib
import numpy as np
import pandas as pd

# Path to CSV file relative to this script
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def find_csv_filename(filename='RingkøbingData.csv'):
    """
    Search for the CSV file in a set of likely repository locations and return
    the first existing absolute path. If not found, raise FileNotFoundError
    listing the attempted locations.

    Candidate locations (checked in order):
      - repository root (../.. from this script)
      - repository `Files/` folder
      - `Plotting/Files/` folder (where the project currently keeps the CSV)
      - the script directory itself
    """
    repo_root = SCRIPT_DIR.parents[1]
    candidates = [
        repo_root / filename,
        repo_root / 'Files' / filename,
        repo_root / 'Plotting' / 'Files' / filename,
        SCRIPT_DIR / filename,
        SCRIPT_DIR.parent / filename,
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    # If none found, raise with helpful message listing attempted paths
    tried = '\n'.join([str(p) for p in candidates])
    raise FileNotFoundError(
        f"Could not find '{filename}'. Tried the following locations:\n{tried}\n"
    )


# Default CSV_FILE_PATH is resolved at load time via find_csv_filename when
# the caller doesn't supply an explicit path.
CSV_FILE_PATH = None

# Number of forecast hours (7 days * 24 hours = 168)
FORECAST_HOURS = 168
USE_CYCLICAL_ENCODING = True  # Set to False to use raw day_of_year and hour values


def format_json_compact_vectors(dataset):
    """
    Format JSON with each vector on a single line, but vectors on separate lines.
    """
    lines = ['{']

    # Format samples array
    lines.append('  "samples": [')
    for i, sample in enumerate(dataset['samples']):
        lines.append('    [')
        for j, vector in enumerate(sample):
            vector_str = json.dumps(vector)
            comma = ',' if j < len(sample) - 1 else ''
            lines.append(f'      {vector_str}{comma}')
        sample_comma = ',' if i < len(dataset['samples']) - 1 else ''
        lines.append(f'    ]{sample_comma}')
    lines.append('  ],')

    # Format targets array (compact)
    lines.append(f'  "targets": {json.dumps(dataset["targets"])},')

    # Format metadata
    lines.append(f'  "feature_names": {json.dumps(dataset["feature_names"])},')
    lines.append(f'  "target_name": {json.dumps(dataset["target_name"])},')
    lines.append(f'  "n_samples": {dataset["n_samples"]},')
    lines.append(f'  "n_forecast_hours": {dataset["n_forecast_hours"]},')
    lines.append(f'  "n_features": {dataset["n_features"]},')
    lines.append(f'  "use_cyclical_encoding": {json.dumps(dataset["use_cyclical_encoding"])}')
    lines.append('}')

    return '\n'.join(lines)


def load_csv(filepath=None):
    """Load the CSV file with proper encoding and delimiter.

    If `filepath` is None, try to locate `RingkøbingData.csv` in likely
    repository locations using `find_csv_filename()`.
    """
    if filepath is None:
        filepath = find_csv_filename('RingkøbingData.csv')

    filepath = pathlib.Path(filepath).resolve()

    # Use comma as delimiter, parse dateTime as datetime
    df = pd.read_csv(
        filepath,
        delimiter=',',
        on_bad_lines='skip',
        low_memory=False,
        encoding='utf-8',
        parse_dates=['dateTime']
    )

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    print(f"Loaded {len(df)} rows from {filepath.name}")
    return df


def create_cyclical_encoding(value, max_value):
    """Create sin/cos cyclical encoding for a single value."""
    angle = 2 * np.pi * value / max_value
    return np.sin(angle), np.cos(angle)


def get_time_features(base_datetime, hour_offset, use_cyclical=True):
    """
    Get date/time features for a specific forecast hour.

    Args:
        base_datetime: The base datetime of the row
        hour_offset: Hours ahead (0-167)
        use_cyclical: If True, use sin/cos encoding. If False, use raw values.

    Returns:
        Tuple of (dateSin/day_of_year, dateCos/days_in_year, timeSin/hour, timeCos/None)
    """
    # Calculate the actual datetime for this forecast hour
    forecast_datetime = base_datetime + pd.Timedelta(hours=hour_offset)

    day_of_year = forecast_datetime.dayofyear
    hour = forecast_datetime.hour
    days_in_year = 366 if forecast_datetime.is_leap_year else 365

    if use_cyclical:
        date_sin, date_cos = create_cyclical_encoding(day_of_year, days_in_year)
        time_sin, time_cos = create_cyclical_encoding(hour, 24)
        return date_sin, date_cos, time_sin, time_cos
    else:
        # Return raw values: day_of_year, hour
        return day_of_year, hour, None, None


def create_forecast_vectors(row, use_cyclical=True):
    """
    Create 168 feature vectors for a single row (one per forecast hour).

    Each vector contains:
        - dateSin, dateCos (or day_of_year if not cyclical)
        - timeSin, timeCos (or hour if not cyclical)
        - cloudCover_X, precipitation_X, relativeHumidity_X, temperature_X, windSpeed_X

    Args:
        row: A pandas Series representing one row from the DataFrame
        use_cyclical: Whether to use cyclical encoding for date/time

    Returns:
        List of 168 feature vectors
    """
    base_datetime = row['dateTime']
    vectors = []

    for hour_offset in range(FORECAST_HOURS):
        # Get time features
        time_features = get_time_features(base_datetime, hour_offset, use_cyclical)

        # Get weather features for this hour
        cloud_cover = row[f'cloudCover_{hour_offset}']
        precipitation = row[f'precipitation_{hour_offset}']
        relative_humidity = row[f'relativeHumidity_{hour_offset}']
        temperature = row[f'temperature_{hour_offset}']
        wind_speed = row[f'windSpeed_{hour_offset}']

        if use_cyclical:
            vector = [
                time_features[0],  # dateSin
                time_features[1],  # dateCos
                time_features[2],  # timeSin
                time_features[3],  # timeCos
                cloud_cover,
                precipitation,
                relative_humidity,
                temperature,
                wind_speed
            ]
        else:
            vector = [
                time_features[0],  # day_of_year
                time_features[1],  # hour
                cloud_cover,
                precipitation,
                relative_humidity,
                temperature,
                wind_speed
            ]

        vectors.append(vector)

    return vectors


def create_dataset(filepath=None, output_path=None, use_cyclical=True, first_row_only=True):
    """
    Extract data from CSV and create feature vectors for each forecast hour.

    Args:
        filepath: Path to the CSV file
        output_path: Path to save the JSON output
        use_cyclical: Whether to use cyclical encoding for date/time
        first_row_only: If True, only process the first row (for testing)

    Returns:
        Dictionary with forecast vectors and metadata, saved as JSON.
    """
    # Load data (locate CSV if filepath not provided)
    df = load_csv(filepath).copy()

    # For testing, only use the first row
    if first_row_only:
        df = df.head(1)
        print(f"\nProcessing first row only (for testing)")

    print(f"\nSample values from first row:")
    print(f"  dateTime: {df['dateTime'].iloc[0]}")
    print(f"  abvaerk: {df['abvaerk'].iloc[0]}")
    print(f"  cloudCover_0: {df['cloudCover_0'].iloc[0]}")
    print(f"  cloudCover_167: {df['cloudCover_167'].iloc[0]}")

    # Create forecast vectors for each row
    all_samples = []
    all_targets = []

    for idx, row in df.iterrows():
        vectors = create_forecast_vectors(row, use_cyclical=use_cyclical)
        all_samples.append(vectors)
        all_targets.append(row['abvaerk'])

    # Define feature names based on encoding type
    if use_cyclical:
        feature_names = ['dateSin', 'dateCos', 'timeSin', 'timeCos',
                         'cloudCover', 'precipitation', 'relativeHumidity',
                         'temperature', 'windSpeed']
    else:
        feature_names = ['day_of_year', 'hour',
                         'cloudCover', 'precipitation', 'relativeHumidity',
                         'temperature', 'windSpeed']

    # Create output dictionary
    dataset = {
        'samples': all_samples,  # Shape: (n_rows, 168, n_features)
        'targets': all_targets,
        'feature_names': feature_names,
        'target_name': 'abvaerk',
        'n_samples': len(all_samples),
        'n_forecast_hours': FORECAST_HOURS,
        'n_features': len(feature_names),
        'use_cyclical_encoding': use_cyclical
    }

    # Save to JSON with compact vector formatting
    if output_path is None:
        output_path = SCRIPT_DIR / 'dataset.json'

    output_path = pathlib.Path(output_path)
    formatted_json = format_json_compact_vectors(dataset)
    with open(output_path, 'w') as f:
        f.write(formatted_json)

    # Count lines in the output file
    line_count = formatted_json.count('\n') + 1

    print(f"\nDataset saved to {output_path}")
    print(f"Lines in file: {line_count}")
    print(f"Shape: ({dataset['n_samples']} samples, {dataset['n_forecast_hours']} hours, {dataset['n_features']} features)")

    return dataset


if __name__ == '__main__':
    # For testing: only first row, with cyclical encoding
    dataset = create_dataset(use_cyclical=USE_CYCLICAL_ENCODING, first_row_only=False)












