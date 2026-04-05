import numpy as np
import pandas as pd
import pwlf
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import logging

try:
    from sklearn.linear_model import HuberRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

@dataclass
class FitterConfig:
    """Configuration for the PowerLawFitter."""
    max_segments: int = 2
    min_segment_len: int = 5
    use_robust: bool = False
    use_weighted: bool = False
    mad_threshold: float = 3.0
    ball_size: str = "Unknown"

class PowerLawFitter:
    """Handles piecewise power-law fitting logic."""

    def __init__(self, config: FitterConfig = FitterConfig()):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Extracts and log-transforms speed and pressure data."""
        speed = df["speed"].values
        pressure = df["pressure"].values
        logV = np.log10(speed)
        logP = np.log10(pressure)
        
        logV_err = df["v_err"].values if "v_err" in df.columns else None
        logP_err = df["p_err"].values if "p_err" in df.columns else None
        
        return logV, logP, logV_err, logP_err

    def remove_outliers(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Removes outliers using initial robust fit and Median Absolute Deviation (MAD)."""
        if not SKLEARN_AVAILABLE:
            return x, y, np.array([])

        huber = HuberRegressor().fit(x.reshape(-1, 1), y)
        pred = huber.predict(x.reshape(-1, 1))
        resid = y - pred
        mad = np.median(np.abs(resid - np.median(resid)))
        mask = np.abs(resid) <= self.config.mad_threshold * mad
        
        outliers_idx = np.where(~mask)[0]
        return x[mask], y[mask], outliers_idx

    def find_optimal_segments(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[int, pwlf.PiecewiseLinFit, np.ndarray]:
        """Finds the optimal number of segments using BIC."""
        bic_list = []
        models = []
        n = len(x)
        
        for k in range(1, self.config.max_segments + 1):
            model = pwlf.PiecewiseLinFit(x, y, weights=weights)
            try:
                breaks = model.fit(k, x_min=x.min(), x_max=x.max())
                
                # Check for minimum segment length
                counts, _ = np.histogram(x, bins=breaks)
                if np.any(counts < self.config.min_segment_len):
                    continue
                
                sse = model.ssr
                p = 2 * k  # slope + intercept per segment
                bic = n * np.log(sse / n) + p * np.log(n)
                bic_list.append(bic)
                models.append((model, breaks))
            except Exception as e:
                self.logger.warning(f"Fitting failed for k={k}: {e}")
                continue

        if not bic_list:
            raise RuntimeError("No valid piecewise model found.")
            
        best_idx = int(np.argmin(bic_list))
        optimal_k = best_idx + 1
        return optimal_k, models[best_idx][0], models[best_idx][1]

    def calculate_metrics(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, y_err: Optional[np.ndarray], p: int) -> Dict[str, float]:
        """Calculates R² and reduced Chi-Squared."""
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        chi2_red = np.nan
        if y_err is not None:
            chi2_red = np.sum(((y - y_pred) / y_err) ** 2) / (len(x) - p)
            
        return {"r2": r2, "chi2_red": chi2_red}

    def run_analysis(self, df: pd.DataFrame) -> Dict:
        """Runs the full analysis pipeline."""
        logV, logP, logV_err, logP_err = self.prepare_data(df)
        
        x_working, y_working = logV, logP
        outliers_idx = np.array([])
        
        if self.config.use_robust:
            x_working, y_working, outliers_idx = self.remove_outliers(logV, logP)
            # Filter outliers in the measurement uncertainties if they exist
            if logP_err is not None:
                # Assume if robust is ON, we filter everything
                mask = np.ones(len(logV), dtype=bool)
                mask[outliers_idx] = False
                logP_err_working = logP_err[mask]
            else:
                logP_err_working = None
        else:
            logP_err_working = logP_err

        weights = 1.0 / logP_err_working if (self.config.use_weighted and logP_err_working is not None) else None
        
        optimal_k, model, breaks = self.find_optimal_segments(x_working, y_working, weights=weights)
        
        # Calculate final robust segments 
        slopes = []
        ses = []
        intercepts = []
        
        # model.calc_slopes() gives slopes
        # model.standard_errors() gives errors
        slopes = model.calc_slopes()
        ses_all = model.standard_errors()
        ses = ses_all[1:]
        intercepts = model.intercepts
        
        # Refit robustly per segment if requested
        if self.config.use_robust and SKLEARN_AVAILABLE:
            final_slopes = []
            final_intercepts = []
            for i in range(len(breaks) - 1):
                mask = (x_working >= breaks[i]) & (x_working <= breaks[i+1])
                Xseg = x_working[mask].reshape(-1, 1)
                Yseg = y_working[mask]
                hub = HuberRegressor().fit(Xseg, Yseg)
                final_slopes.append(hub.coef_[0])
                final_intercepts.append(hub.intercept_)
            slopes = np.array(final_slopes)
            intercepts = np.array(final_intercepts)

        # Calculate metrics per segment
        y_hat = model.predict(x_working)
        segment_metrics = []
        for i in range(len(breaks) - 1):
            mask = (x_working >= breaks[i]) & (x_working <= breaks[i+1])
            metrics = self.calculate_metrics(
                x_working[mask], 
                y_working[mask], 
                y_hat[mask], 
                logP_err_working[mask] if logP_err_working is not None else None,
                2 # slope + intercept
            )
            segment_metrics.append(metrics)

        return {
            "logV": logV,
            "logP": logP,
            "logV_err": logV_err,
            "logP_err": logP_err,
            "outliers_idx": outliers_idx,
            "optimal_k": optimal_k,
            "breaks": breaks,
            "slopes": slopes,
            "ses": ses,
            "intercepts": intercepts,
            "segment_metrics": segment_metrics,
            "y_hat": y_hat,
            "x_working": x_working,
            "y_working": y_working
        }
