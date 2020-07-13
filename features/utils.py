import csv
import os
import pickle
from typing import List

import numpy as np


def save_features(path: str, features: List[np.ndarray]):
    """
    Saves features to a compressed .npz file with array name "features".

    :param path: Path to a .npz file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")
    - .npz (assumes features are saved with name "features")
    - .npy
    - .csv/.txt (assumes comma-separated features with a header and with one line per molecule)
    - .pkl/.pckl/.pickle containing a sparse numpy array (TODO: remove this option once we are no longer dependent on it)

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    elif extension == '.npy':
        features = np.load(path)
    elif extension in ['.csv', '.txt']:
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            features = np.array([[float(value) for value in row] for row in reader])
    elif extension in ['.pkl', '.pckl', '.pickle']:
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


def read_sdf(path, n=None, keepHs=False):

    with open(path) as f:
        lines = f.readlines()

    # separate into individual sdfs
    size = len(lines)
    idx_lines = [idx + 1 for idx, line in enumerate(lines) if line.startswith('$$$$')]
    res = [lines[i: j] for i, j in zip([0] + idx_lines, idx_lines + ([size] if idx_lines[-1] != size else []))]

    confs = []
    for sdf in res[:n]:

        smiles = sdf[0].rstrip()
        properties = sdf[2].split()
        # first three lines header
        del sdf[:3]

        L1 = sdf.pop(0).split()
        natoms = int(L1[0])
        positions = []
        symbols = []
        for line in sdf[:natoms]:
            x, y, z, symbol = line.split()[:4]
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])

        if not keepHs:
            positions = [p for i, p in enumerate(positions) if symbols[i] != 'H']
            symbols = [s for s in symbols if s != 'H']

        confs.append((positions, symbols))

    return confs
