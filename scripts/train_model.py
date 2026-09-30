"""
scripts/train_model.py
======================
Production ML training pipeline for EstateIQ.

This script:
1. Downloads the Ames Housing dataset (real property data)
2. Engineers features matching the API schema
3. Trains a Gradient Boosting model
4. Evaluates with proper metrics
5. Saves the model and metrics for the app to use
6. Generates SHAP values for explainability

Run with: python scripts/train_model.py
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    shap = None  # type: ignore
    _SHAP_AVAILABLE = False
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────

MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "house_price_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
SHAP_PATH    = os.path.join(MODEL_DIR, "shap_values.json")
PLOTS_DIR    = os.path.join("static", "plots")

# Ames Housing dataset — free, real, well-known in ML community
DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "datasciencedojo/datasets/master/AmesHousing.csv"
)

# Features we use — mapped from Ames column names to our API names
FEATURE_MAP = {
    "Gr Liv Area":    "area",
    "Bedroom AbvGr":  "bedrooms",
    "Full Bath":      "bathrooms",
    "TotRms AbvGrd":  "floors",        # repurposed as room count proxy
    "Year Built":     "year_built",
    "Garage Cars":    "garage",
    "Pool Area":      "pool",
    "Lot Area":       "garden",
    "Overall Qual":   "location_score",
    "Overall Cond":   "condition",
}

TARGET = "SalePrice"

MODEL_PARAMS = {
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        5,
    "min_samples_leaf": 10,
    "subsample":        0.8,
    "random_state":     42,
    "loss":             "huber",  # robust to outliers
}


# ─────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load the Ames Housing dataset.
    Falls back to local file if network is unavailable.
    """
    local_path = os.path.join(MODEL_DIR, "ames_housing.csv")

    if os.path.exists(local_path):
        logger.info(f"Loading dataset from local cache: {local_path}")
        return pd.read_csv(local_path)

    logger.info("Downloading Ames Housing dataset...")
    try:
        df = pd.read_csv(DATASET_URL)
        os.makedirs(MODEL_DIR, exist_ok=True)
        df.to_csv(local_path, index=False)
        logger.info(f"Dataset cached at: {local_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.info("Generating synthetic dataset for demonstration...")
        return generate_synthetic_data()


def generate_synthetic_data(n: int = 2000) -> pd.DataFrame:
    """
    Fallback synthetic dataset if network is unavailable.
    Uses realistic distributions based on Ames Housing statistics.
    """
    np.random.seed(42)
    data = {
        "Gr Liv Area":    np.random.normal(1500, 500, n).clip(400, 5000),
        "Bedroom AbvGr":  np.random.randint(1, 7, n),
        "Full Bath":      np.random.randint(1, 5, n),
        "TotRms AbvGrd":  np.random.randint(4, 14, n),
        "Year Built":     np.random.randint(1900, 2023, n),
        "Garage Cars":    np.random.randint(0, 4, n),
        "Pool Area":      np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "Lot Area":       np.random.normal(10000, 5000, n).clip(1000, 50000),
        "Overall Qual":   np.random.randint(1, 11, n),
        "Overall Cond":   np.random.randint(1, 10, n),
    }
    # Generate realistic prices correlated with features
    df = pd.DataFrame(data)
    price = (
        df["Gr Liv Area"]    * 80
        + df["Overall Qual"] * 15000
        + df["Year Built"]   * 100
        + df["Garage Cars"]  * 8000
        + df["Full Bath"]    * 5000
        + np.random.normal(0, 20000, n)
    )
    df["SalePrice"] = price.clip(50000, 800000)
    return df


# ─────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Select and transform features for the model.
    Returns X (features) and y (target).
    """
    # Select only columns we need
    cols = list(FEATURE_MAP.keys()) + [TARGET]
    available = [c for c in cols if c in df.columns]

    if TARGET not in available:
        raise ValueError(f"Target column '{TARGET}' not found in dataset")

    df_clean = df[available].copy()

    # Fill missing values with median (robust to outliers)
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Binarize pool (Ames has pool area in sqft, we want 0/1)
    if "Pool Area" in df_clean.columns:
        df_clean["Pool Area"] = (df_clean["Pool Area"] > 0).astype(int)

    # Binarize garden (lot area normalized to 0-1 scale)
    if "Lot Area" in df_clean.columns:
        max_lot = df_clean["Lot Area"].quantile(0.95)
        df_clean["Lot Area"] = (df_clean["Lot Area"] / max_lot).clip(0, 1)

    # Binarize garage (0 = no garage, 1+ = has garage)
    if "Garage Cars" in df_clean.columns:
        df_clean["Garage Cars"] = (df_clean["Garage Cars"] > 0).astype(int)

    # Normalize location score to 1-10
    if "Overall Qual" in df_clean.columns:
        df_clean["Overall Qual"] = df_clean["Overall Qual"].clip(1, 10)

    # Normalize condition to 1-10
    if "Overall Cond" in df_clean.columns:
        df_clean["Overall Cond"] = df_clean["Overall Cond"].clip(1, 10)

    # Rename to our API schema names
    feature_cols = [c for c in FEATURE_MAP.keys() if c in df_clean.columns]
    X = df_clean[feature_cols].rename(columns=FEATURE_MAP)
    y = df_clean[TARGET]

    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Price range: ${y.min():,.0f} — ${y.max():,.0f}")

    return X, y


# ─────────────────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train the Gradient Boosting model with cross-validation.
    Returns trained pipeline and train/test splits.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)}")

    # Pipeline ensures scaler is fitted on train only (no data leakage)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(**MODEL_PARAMS))
    ])

    logger.info("Training model... (this takes 1-2 minutes)")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")

    return pipeline, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────

def evaluate_model(pipeline, X_train, X_test, y_train, y_test) -> dict:
    """
    Comprehensive model evaluation with multiple metrics.
    """
    y_pred = pipeline.predict(X_test)

    # Core metrics
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # Cross-validation on training set
    logger.info("Running 5-fold cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring="r2", n_jobs=-1
    )

    metrics = {
        "r2_score":       round(float(r2), 4),
        "mae":            round(float(mae), 2),
        "rmse":           round(float(rmse), 2),
        "mape":           round(float(mape), 2),
        "accuracy":       round(float(r2) * 100, 1),
        "cv_mean":        round(float(cv_scores.mean()), 4),
        "cv_std":         round(float(cv_scores.std()), 4),
        "train_size":     len(X_train),
        "test_size":      len(X_test),
        "features":       len(X_train.columns),
        "feature_names":  list(X_train.columns),
        "algorithm":      "Gradient Boosting Regressor",
        "sklearn_version": __import__("sklearn").__version__,
    }

    logger.info(f"R² Score:     {r2:.4f} ({r2*100:.1f}%)")
    logger.info(f"MAE:          ${mae:,.0f}")
    logger.info(f"RMSE:         ${rmse:,.0f}")
    logger.info(f"MAPE:         {mape:.1f}%")
    logger.info(f"CV Score:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return metrics


# ─────────────────────────────────────────────────────────
#  SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────


def compute_shap(pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """
    Compute SHAP values for model explainability.

    SHAP (SHapley Additive exPlanations) tells you exactly how much
    each feature contributed to each prediction.
    """
    logger.info("Computing SHAP values...")

    # Get pipeline components
    model = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]

    # Scale data
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    # Use smaller sample for speed
    sample_size = min(200, len(X_train_scaled))
    X_sample = X_train_scaled.sample(sample_size, random_state=42)

    try:
        if not _SHAP_AVAILABLE:
            raise ImportError("shap not available")
        # SHAP explainer
        explainer = shap.TreeExplainer(model)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values_array = np.array(shap_values[0])
        else:
            shap_values_array = np.array(shap_values)

        # Feature importance
        feature_importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": np.abs(shap_values_array).mean(axis=0)
        }).sort_values("importance", ascending=False)

        # FIX: expected_value may be array/list
        expected_value = explainer.expected_value

        if isinstance(expected_value, (list, np.ndarray)):
            base_value = float(np.array(expected_value).flatten()[0])
        else:
            base_value = float(expected_value)

        shap_data = {
            "feature_importance": feature_importance.to_dict("records"),
            "base_value": base_value,
            "feature_names": list(X_train.columns),
            "sample_shap_values": shap_values_array[:5].tolist(),
        }

        logger.info("Top feature importances by SHAP:")

        for _, row in feature_importance.iterrows():
            logger.info(
                f"  {row['feature']:20s}: {row['importance']:,.0f}"
            )

        # Save SHAP plot
        os.makedirs(PLOTS_DIR, exist_ok=True)

        try:
            plt.figure(figsize=(10, 6))

            shap.summary_plot(
                shap_values_array,
                X_sample,
                plot_type="bar",
                show=False
            )

            plt.title("Feature Importance (SHAP Values)")
            plt.tight_layout()

            plot_path = os.path.join(
                PLOTS_DIR,
                "shap_importance.png"
            )

            plt.savefig(
                plot_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close()

            logger.info(f"SHAP plot saved to {plot_path}")

        except Exception as e:
            logger.warning(f"Could not save SHAP plot: {e}")

        return shap_data

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")

        # Fallback feature importance from model
        try:
            importance = model.feature_importances_

            feature_importance = pd.DataFrame({
                "feature": X_train.columns,
                "importance": importance
            }).sort_values("importance", ascending=False)

            return {
                "feature_importance": feature_importance.to_dict("records"),
                "base_value": 0.0,
                "feature_names": list(X_train.columns),
                "sample_shap_values": [],
            }

        except Exception:
            return {
                "feature_importance": [],
                "base_value": 0.0,
                "feature_names": list(X_train.columns),
                "sample_shap_values": [],
            }

# ─────────────────────────────────────────────────────────
#  SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────

def save_artifacts(pipeline, metrics: dict, shap_data: dict):
    """Save model, metrics, and SHAP data for the app to use."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save trained model
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Model saved: {MODEL_PATH}")

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {METRICS_PATH}")

    # Save SHAP data
    with open(SHAP_PATH, "w") as f:
        json.dump(shap_data, f, indent=2)
    logger.info(f"SHAP data saved: {SHAP_PATH}")


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("EstateIQ — ML Training Pipeline")
    logger.info("=" * 60)

    # Load data
    df = load_data()
    logger.info(f"Loaded {len(df)} records")

    # Engineer features
    X, y = engineer_features(df)

    # Train model
    pipeline, X_train, X_test, y_train, y_test = train_model(X, y)

    # Evaluate
    metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)

    # SHAP explainability
    shap_data = compute_shap(pipeline, X_train, X_test)

    # Save everything
    save_artifacts(pipeline, metrics, shap_data)

    logger.info("=" * 60)
    logger.info("Training complete. Summary:")
    logger.info(f"  R² Score:  {metrics['r2_score']} ({metrics['accuracy']}%)")
    logger.info(f"  MAE:       ${metrics['mae']:,.0f}")
    logger.info(f"  RMSE:      ${metrics['rmse']:,.0f}")
    logger.info(f"  Model:     {MODEL_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()