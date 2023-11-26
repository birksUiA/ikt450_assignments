import numpy as np
import os
import torch
import torch.nn as nn


def parse_ecg_file(file_path: str) -> list[float]:
    """Parse an ECG file to extract the desired readings.

    Args:
        file_path (str): Path to the ECG data file.

    Returns:
        list of float: Extracted data from the ECG file.
    """
    with open(file_path, "r") as file:
        data = [
            float(row.strip().split("   ")[1])
            for row in file.read().strip().split("\n")
        ]
    return data


def zero_pad_if_required(data: list[float], n: int) -> list[float]:
    """Pad or truncate data to a specified length.

    Args:
        data (list): List of data to be padded or truncated.
        n (int): Desired length of the output list.

    Returns:
        list: Padded or truncated data list.
    """
    if len(data) < n:
        return data + [0] * (n - len(data))
    elif len(data) > n:
        return data[:n]
    else:
        return data


def load_ecg_data(directory_path: str, measurements: int = 75) -> list[np.ndarray]:
    """Load ECG data from a directory and return it as numpy arrays.

    Args:
        directory_path (str): Path to the directory containing ECG data.
        measurements (int, optional): Number of measurements per lead. Defaults to 75.

    Returns:
        list: Tuple containing features array and labels array.
    """
    features = []
    labels = []

    def get_lead_pair_data(directory: str, label: int) -> None:
        """Fetch paired lead data from a directory and append to features and labels."""
        for file_name in os.listdir(directory):
            if file_name.endswith(".0"):
                paired_file_name = file_name[:-2] + ".1"
                if paired_file_name in os.listdir(directory):
                    lead_0_path = os.path.join(directory, file_name)
                    lead_1_path = os.path.join(directory, paired_file_name)
                    lead_0 = parse_ecg_file(lead_0_path)
                    lead_1 = parse_ecg_file(lead_1_path)
                    features.append(
                        zero_pad_if_required(lead_0, measurements)
                    )
                    labels.append(label)

    get_lead_pair_data(os.path.join(directory_path, "normal"), 0)
    get_lead_pair_data(os.path.join(directory_path, "abnormal"), 1)

    # Convert features and labels to numpy arrays
    features_array = np.array(features)
    labels_array = np.array(labels)

    return [features_array, labels_array]


def split_data(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data into train, validation, and test sets."""
    total_data = len(features)
    train_end = int(total_data * train_ratio)
    val_end = int(total_data * (train_ratio + val_ratio))

    x_train = features[:train_end]
    y_train = labels[:train_end]

    x_val = features[train_end:val_end]
    y_val = labels[train_end:val_end]

    x_test = features[val_end:]
    y_test = labels[val_end:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def convert_to_tensors(*arrays: np.ndarray) -> list[torch.FloatTensor]:
    """Convert a list of numpy arrays to tensors."""
    return [torch.FloatTensor(arr) for arr in arrays]
