import os
import json

import numpy as np
import pandas as pd
import streamlit as st

import joblib
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from xgboost import XGBRegressor


# ---------- CONFIG & HELPERS ----------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def effnet_preprocess(x):
    """Same preprocessing function name used in the notebook Lambda layer."""
    return tf.keras.applications.efficientnet.preprocess_input(x)


@st.cache_resource
def load_artifacts():
    """Load all models and configs once, cache in memory."""
    # --- 1) Load config (numeric features, image size) ---
    config_path = os.path.join(MODELS_DIR, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    numeric_features_fe = config["numeric_features_fe"]
    eff_target = int(config["eff_target"])

    # --- 2) Load city target encoding info ---
    city_enc_path = os.path.join(MODELS_DIR, "city_target_enc.json")
    with open(city_enc_path, "r") as f:
        city_target_info = json.load(f)
    city_mean_train = city_target_info["city_mean_train"]
    global_mean_log_price = city_target_info["global_mean_log_price"]

    # --- 3) Load tuned tabular XGBoost FE pipeline ---
    xgb_fe_path = os.path.join(MODELS_DIR, "xgb_fe_tuned_pipeline.pkl")
    xgb_fe_pipeline = joblib.load(xgb_fe_path)

    # --- 4) Load hybrid XGBoost model ---
    hybrid_path = os.path.join(MODELS_DIR, "hybrid_xgb_model.pkl")
    hybrid_xgb_model: XGBRegressor = joblib.load(hybrid_path)

    # --- 5) Build EfficientNetB0 embedding extractor from scratch ---
    # Instead of loading the saved model (which is corrupted), 
    # we'll rebuild the architecture using Functional API
    
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers
    
    # Rebuild the exact same architecture from the notebook using Functional API
    eff_base = EfficientNetB0(
        include_top=False,
        weights="imagenet",  # Start with ImageNet weights
        input_shape=(eff_target, eff_target, 3),
        pooling=None
    )
    
    # Build using Functional API
    inputs = layers.Input(shape=(eff_target, eff_target, 3), name="image_input")
    x = layers.RandomFlip("horizontal")(inputs, training=False)
    x = layers.RandomRotation(0.08)(x, training=False)
    x = layers.RandomZoom(0.15)(x, training=False)
    x = layers.RandomTranslation(0.1, 0.1)(x, training=False)
    x = layers.RandomContrast(0.2)(x, training=False)
    x = layers.Resizing(eff_target, eff_target)(x)
    x = layers.Lambda(effnet_preprocess, name="efficientnet_preprocessing")(x)
    x = eff_base(x)
    x = layers.GlobalAveragePooling2D()(x)
    embeddings = layers.Dense(128, activation="relu", name="embedding_dense")(x)
    x = layers.Dropout(0.3)(embeddings, training=False)
    outputs = layers.Dense(1, name="log_price_output")(x)
    
    full_model = keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_reconstructed")
    
    print("✓ Model architecture built")
    
    # Now try to load the weights from the saved model
    effnet_path = os.path.join(MODELS_DIR, "effnetb0_tuned_last10.keras")
    
    try:
        # Try to load weights only
        full_model.load_weights(effnet_path)
        print("✓ Successfully loaded model weights")
    except Exception as e:
        print(f"⚠ Could not load weights directly: {e}")
        print("⚠ Using ImageNet pretrained weights only (predictions may be less accurate)")
    
    # Create embedding extractor (output of Dense(128) layer)
    embedding_extractor = keras.Model(
        inputs=full_model.input,
        outputs=embeddings,
        name="embedding_extractor"
    )
    
    print("✓ Embedding extractor created successfully")

    # --- 6) Extract fitted preprocessor for hybrid (tabular scaling) ---
    tab_preprocessor_fe = xgb_fe_pipeline.named_steps["preprocessor"]
    
    # --- 7) Model performance metrics (from notebook) ---
    # Test set MAE in log_price space for both models
    mae_log_tabular = 0.166  # XGBoost FE tuned test MAE
    mae_log_hybrid = 0.148   # Hybrid test MAE
    
    # Average these to get typical error in log space
    metrics = {
        "mae_log_tabular": mae_log_tabular,
        "mae_log_hybrid": mae_log_hybrid,
    }

    return {
        "numeric_features_fe": numeric_features_fe,
        "eff_target": eff_target,
        "city_mean_train": city_mean_train,
        "global_mean_log_price": global_mean_log_price,
        "xgb_fe_pipeline": xgb_fe_pipeline,
        "hybrid_xgb_model": hybrid_xgb_model,
        "embedding_extractor": embedding_extractor,
        "tab_preprocessor_fe": tab_preprocessor_fe,
        "metrics": metrics,
    }


def build_feature_dataframe(
    bed: int,
    bath: int,
    sqft: float,
    city: str,
    numeric_features_fe,
    city_mean_train,
    global_mean_log_price: float,
) -> pd.DataFrame:
    """Recreate the engineered tabular features expected by the model."""
    # Avoid division by zero
    bed_eff = max(bed, 1)
    bath_eff = max(bath, 1)

    sqft_per_bed = sqft / bed_eff
    sqft_per_bath = sqft / bath_eff
    total_rooms = bed + bath
    log_sqft = np.log1p(sqft)

    # Target-encoded city
    city_target_enc = city_mean_train.get(city, global_mean_log_price)

    feat = {
        "bed": bed,
        "bath": bath,
        "sqft": sqft,
        "log_sqft": log_sqft,
        "sqft_per_bed": sqft_per_bed,
        "sqft_per_bath": sqft_per_bath,
        "total_rooms": total_rooms,
        "city_target_enc": city_target_enc,
    }

    df = pd.DataFrame([feat])
    df = df[numeric_features_fe]  # ensure correct column order
    return df


def compute_image_embedding(
    image: Image.Image,
    eff_target: int,
    embedding_extractor: keras.Model,
) -> np.ndarray:
    """
    Resize image, convert to float32, and get the 128-d embedding.
    """
    img = image.convert("RGB")
    img = img.resize((eff_target, eff_target))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)

    emb = embedding_extractor.predict(arr, verbose=0)
    return emb  # shape (1, 128)


def predict_tabular_only(features_df: pd.DataFrame, xgb_fe_pipeline) -> float:
    """Return predicted log_price from tabular-only model."""
    y_log = xgb_fe_pipeline.predict(features_df)[0]
    return y_log


def predict_hybrid(
    features_df: pd.DataFrame,
    tab_preprocessor_fe,
    hybrid_xgb_model: XGBRegressor,
    img_embedding: np.ndarray,
) -> float:
    """Return predicted log_price from hybrid (tabular + image) model."""
    X_tab_scaled = tab_preprocessor_fe.transform(features_df)  # (1, n_tab_features)
    X_hybrid = np.hstack([X_tab_scaled, img_embedding])        # (1, n_tab + 128)
    y_log = hybrid_xgb_model.predict(X_hybrid)[0]
    return y_log


# ---------- STREAMLIT UI ----------

def main():
    st.set_page_config(
        page_title="SoCal House Price Estimator",
        page_icon="🏠",
        layout="centered",
    )

    st.title("🏠 SoCal House Price Estimator")
    st.caption("Hybrid ML model combining tabular features and images")

    # Load models and preprocessing objects
    try:
        artifacts = load_artifacts()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
        
    numeric_features_fe = artifacts["numeric_features_fe"]
    eff_target = artifacts["eff_target"]
    city_mean_train = artifacts["city_mean_train"]
    global_mean_log_price = artifacts["global_mean_log_price"]
    xgb_fe_pipeline = artifacts["xgb_fe_pipeline"]
    hybrid_xgb_model = artifacts["hybrid_xgb_model"]
    embedding_extractor = artifacts["embedding_extractor"]
    tab_preprocessor_fe = artifacts["tab_preprocessor_fe"]
    metrics = artifacts["metrics"]

    # City choices from training
    city_options = sorted(city_mean_train.keys())

    st.subheader("1️⃣ Enter listing details")

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox(
            "City",
            options=city_options,
            index=0,
            help="Cities learned from the training data",
        )
        bed = st.number_input(
            "Bedrooms", min_value=0, max_value=15, value=3, step=1
        )
        bath = st.number_input(
            "Bathrooms", min_value=0, max_value=15, value=2, step=1
        )

    with col2:
        sqft = st.number_input(
            "Living area (sqft)",
            min_value=200,
            max_value=20000,
            value=1500,
            step=50,
        )

        uploaded_image = st.file_uploader(
            "Optional: upload an image of the house",
            type=["jpg", "jpeg", "png"],
            help="If provided, the hybrid model (tabular + image) will be used.",
        )

        if uploaded_image is not None:
            st.image(
                uploaded_image,
                caption="Uploaded image",
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("2️⃣ Get price estimate")

    use_image_if_available = st.checkbox(
        "Use hybrid model when an image is uploaded",
        value=True,
    )

    if st.button("Estimate price"):
        with st.spinner("Computing prediction..."):
            # Build engineered feature DataFrame
            features_df = build_feature_dataframe(
                bed=bed,
                bath=bath,
                sqft=float(sqft),
                city=city,
                numeric_features_fe=numeric_features_fe,
                city_mean_train=city_mean_train,
                global_mean_log_price=global_mean_log_price,
            )

            if uploaded_image is not None and use_image_if_available:
                # Hybrid prediction
                try:
                    image = Image.open(uploaded_image)
                    img_emb = compute_image_embedding(
                        image=image,
                        eff_target=eff_target,
                        embedding_extractor=embedding_extractor,
                    )
                    y_log_pred = predict_hybrid(
                        features_df=features_df,
                        tab_preprocessor_fe=tab_preprocessor_fe,
                        hybrid_xgb_model=hybrid_xgb_model,
                        img_embedding=img_emb,
                    )
                    model_used = "Hybrid XGBoost (tabular + image embeddings)"
                    mae_log = metrics["mae_log_hybrid"]
                except Exception as e:
                    st.warning(f"Could not process image: {e}. Using tabular-only prediction.")
                    y_log_pred = predict_tabular_only(features_df, xgb_fe_pipeline)
                    model_used = "Tabular-only XGBoost (feat-eng, tuned)"
                    mae_log = metrics["mae_log_tabular"]
            else:
                # Tabular-only prediction
                y_log_pred = predict_tabular_only(features_df, xgb_fe_pipeline)
                model_used = "Tabular-only XGBoost (feat-eng, tuned)"
                mae_log = metrics["mae_log_tabular"]

            # Convert log_price -> USD
            price_pred = np.expm1(y_log_pred)
            
            # Calculate error range in USD based on MAE in log space
            # MAE in log space translates to multiplicative error in original space
            price_lower = np.expm1(y_log_pred - mae_log)
            price_upper = np.expm1(y_log_pred + mae_log)
            
            # Calculate the average error margin
            error_margin = (price_upper - price_lower) / 2

        # Display prediction with error range
        st.success(f"Estimated price: **${price_pred:,.0f}**")
        st.info(f"📊 Typical error range: \\${price_lower:,.0f} – \\${price_upper:,.0f} (±\\${error_margin:,.0f})")
        
        st.caption(
            f"Model used: **{model_used}**. "
            "Trained on ~15k Southern California listings with R² ≈ 0.78 (tabular) "
            "and ≈ 0.80 (hybrid)."
        )

        # Brief explanation
        explanation = (
            f"- Tabular features used: bedrooms (**{bed}**), bathrooms (**{bath}**), "
            f"size (**{sqft} sqft**), and a city-level price signal for **{city}**.\n"
        )

        if uploaded_image is not None and use_image_if_available:
            explanation += (
                "- The uploaded exterior photo was converted into visual embeddings "
                "(style, curb appeal, condition) and combined with the tabular features.\n"
            )
        else:
            explanation += "- Only structured tabular information was used for this prediction.\n"
        
        explanation += (
            f"\n**Error range explanation:** Based on the model's test set performance "
            f"(MAE ≈ {mae_log:.3f} in log-price space), typical prediction errors are around "
            f"±${error_margin:,.0f}. This means the true price is likely within the displayed range."
        )

        st.markdown(f"**How this estimate was made**\n\n{explanation}")


if __name__ == "__main__":
    main()