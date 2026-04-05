import os
import glob
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
from scipy.stats import linregress
from scipy.signal import savgol_filter
from PIL import Image

# Use robust regression if scikit-learn is available
try:
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available: robust regression will be disabled.")

@dataclass
class TrackerConfig:
    """Configuration for the BallTracker."""
    pixel_to_cm: float = 108.3
    strictness: float = 1.0
    use_robust: bool = True
    ransac_threshold: Optional[float] = None  # if None, uses default based on strictness
    smooth_window: int = 5
    smooth_polyorder: int = 2

class BallTracker:
    """Handles image processing and ball tracking logic."""

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def extract_epoch_from_filename(filename: str) -> Optional[int]:
        """Extracts the hexadecimal timestamp from 'prefix_<hex>.<ext>'."""
        base = os.path.splitext(os.path.basename(filename))[0]
        if "_" in base:
            _, hex_part = base.split("_", 1)
            try:
                return int(hex_part, 16)
            except ValueError:
                return None
        return None

    def process_images(self, folder_path: str) -> List[Tuple[str, Optional[float], int]]:
        """Processes TIFF images in a folder to detect ball centroids."""
        paths = glob.glob(os.path.join(folder_path, "*.tif")) + glob.glob(
            os.path.join(folder_path, "*.tiff")
        )
        valid = []
        for p in paths:
            e = self.extract_epoch_from_filename(p)
            if e is not None:
                valid.append((p, e))
        
        valid.sort(key=lambda x: x[1])

        results = []
        for path, epoch in valid:
            try:
                img = Image.open(path).convert("L")
                arr = np.array(img, dtype=float)
                inv = 255.0 - arr

                # 2D intensity-weighted centroid detection
                mean_val = inv.mean()
                std_val = inv.std()
                thresh = mean_val + 1.96 * std_val
                mask = inv > thresh

                if np.any(mask):
                    y_idx, x_idx = np.nonzero(mask)
                    values = inv[y_idx, x_idx]
                    x_centroid = float(np.sum(x_idx * values) / np.sum(values))
                else:
                    self.logger.warning(f"No bright region detected in {path}")
                    x_centroid = None
                
                results.append((path, x_centroid, epoch))
            except Exception as e:
                self.logger.error(f"Error processing {path}: {e}")
                
        return results

    def calculate_speed_fit(
        self, 
        detections: List[Tuple[str, Optional[float], int]]
    ) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
        """Computes regression and smoothing on detected ball positions."""
        epochs = [e for _, _, e in detections]
        if not epochs:
            raise ValueError("No valid epochs for time.")
        t0 = min(epochs)

        times, positions = [], []
        for _, x, e in detections:
            if x is not None:
                times.append((e - t0) * 1e-3)
                positions.append(x / self.config.pixel_to_cm)

        if len(times) < 2:
            raise ValueError("Not enough data points for regression.")

        times_arr = np.array(times)
        pos_arr = np.array(positions)

        # Apply temporal smoothing
        window = self.config.smooth_window
        if len(pos_arr) >= window and window % 2 == 1:
            pos_smooth = savgol_filter(
                pos_arr, window_length=window, polyorder=self.config.smooth_polyorder
            )
        else:
            pos_smooth = pos_arr

        X = times_arr.reshape(-1, 1)
        y = pos_smooth

        # Robust regression with RANSAC
        if self.config.use_robust and SKLEARN_AVAILABLE:
            base = LinearRegression()
            thresh = (
                self.config.ransac_threshold
                if self.config.ransac_threshold is not None
                else np.std(y) * self.config.strictness
            )
            ransac = RANSACRegressor(
                estimator=base, residual_threshold=thresh, random_state=0
            )
            ransac.fit(X, y)
            inliers = ransac.inlier_mask_
            lr = linregress(times_arr[inliers], y[inliers])
        else:
            inliers = np.ones_like(times_arr, dtype=bool)
            lr = linregress(times_arr, y)

        fit = {
            "slope": lr.slope,
            "intercept": lr.intercept,
            "r_value": lr.rvalue,
            "p_value": lr.pvalue,
            "stderr": lr.stderr,
        }
        return times_arr, pos_arr, fit, inliers
