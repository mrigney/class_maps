"""Terrain profile save/load: persist trained models and class definitions.

Profiles are saved as .cmp files (ZIP archives containing JSON metadata
and pickled sklearn model/scaler).
"""

import json
import os
import pickle
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class TerrainProfile:
    """Container for a terrain classification profile."""
    class_definitions: Dict[int, dict]
    model: Any  # sklearn RandomForestClassifier
    scaler: Any  # sklearn StandardScaler
    slic_params: Dict[str, float]
    metadata: Dict[str, str] = field(default_factory=dict)


def save_profile(path, profile):
    """Save a terrain profile to a .cmp file.

    Parameters
    ----------
    path : str
        Output file path (should end in .cmp).
    profile : TerrainProfile
        The profile to save.
    """
    metadata = {
        "version": 1,
        "created": datetime.now().isoformat(),
        "class_definitions": {
            str(k): v for k, v in profile.class_definitions.items()
        },
        "slic_params": profile.slic_params,
        "metadata": profile.metadata,
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Save metadata as JSON
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        # Save model and scaler as pickle
        model_bytes = pickle.dumps(profile.model)
        zf.writestr("model.pkl", model_bytes)

        scaler_bytes = pickle.dumps(profile.scaler)
        zf.writestr("scaler.pkl", scaler_bytes)


def load_profile(path):
    """Load a terrain profile from a .cmp file.

    Parameters
    ----------
    path : str
        Path to the .cmp file.

    Returns
    -------
    TerrainProfile
        The loaded profile.
    """
    with zipfile.ZipFile(path, "r") as zf:
        metadata = json.loads(zf.read("metadata.json"))
        model = pickle.loads(zf.read("model.pkl"))
        scaler = pickle.loads(zf.read("scaler.pkl"))

    # Convert class definition keys back to int
    class_defs = {}
    for k, v in metadata["class_definitions"].items():
        class_id = int(k)
        # Ensure color is a tuple
        v["color"] = tuple(v["color"])
        class_defs[class_id] = v

    return TerrainProfile(
        class_definitions=class_defs,
        model=model,
        scaler=scaler,
        slic_params=metadata.get("slic_params", {}),
        metadata=metadata.get("metadata", {}),
    )
