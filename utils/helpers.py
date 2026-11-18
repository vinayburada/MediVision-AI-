"""
MediVision AI - General Helper Utilities
UUID formatting, base64 converters, date/time formatting, and filename sanitization.
"""
from pathlib import Path
import uuid
import base64
import re
from datetime import datetime, timezone
from typing import Union
import cv2
import numpy as np


def generate_analysis_id() -> str:
    """Generate professional medical identifier: MVA-YYYYMMDD-XXXX."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    return f"MVA-{date_str}-{short_uuid}"


def sanitize_filename(filename: str) -> str:
    """Strip dangerous characters from uploaded filenames."""
    name = Path(filename).name
    # Keep alphanumeric, dot, dash, underscore
    cleaned = re.sub(r"[^a-zA-Z0-9_\.-]", "_", name)
    return cleaned[:100]


def image_to_base64(image_path_or_array: Union[str, Path, np.ndarray]) -> str:
    """Convert an image file or numpy array into base64 data string."""
    if isinstance(image_path_or_array, (str, Path)):
        with open(image_path_or_array, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    elif isinstance(image_path_or_array, np.ndarray):
        # Array in RGB -> BGR for opencv encoding
        if image_path_or_array.ndim == 3 and image_path_or_array.shape[2] == 3:
            bgr = cv2.cvtColor(image_path_or_array, cv2.COLOR_RGB2BGR)
        else:
            bgr = image_path_or_array
        _, buffer = cv2.imencode(".png", bgr)
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    raise TypeError("Expected file path or numpy array.")


def format_timestamp(iso_str: str) -> str:
    """Convert ISO UTC string to human-readable format."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y - %H:%M:%S UTC")
    except Exception:
        return iso_str
