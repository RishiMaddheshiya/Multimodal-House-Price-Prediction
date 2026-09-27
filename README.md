# 🏠 Hybrid House Price Prediction (Tabular + Images)

A multi-modal ML system combining engineered tabular features with CNN image embeddings.
This project builds a hybrid machine learning model that predicts house prices by combining:
- **Structured tabular data** (bedrooms, bathrooms, square footage, city…)
- **Visual information** extracted from exterior house photos using EfficientNet
- **Gradient-boosted decision trees** (XGBoost) for final price prediction

The approach outperforms tabular-only models by capturing implicit visual attributes such as condition, curb appeal, architectural style, landscaping, and exterior quality—features that are typically unavailable or too subjective for users to input manually.

## 📂 Dataset

The project uses the public Kaggle dataset:

🔗 [House Prices and Images - SoCal](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal)

It contains:
- **15,474** Southern California listings
- Tabular metadata (bed/bath/sqft, city, price, etc.)
- One exterior house image per listing
- Clean structure with no missing values

The dataset is not included in the repo. Download it from Kaggle and place it like this:
```
data/
├── socal2.csv
└── socal_pics/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

---

## ✨ Project Highlights

### ✔️ Advanced Feature Engineering
- `log₁₊price` target transformation
- Spaciousness metrics: `sqft_per_bed`, `sqft_per_bath`
- Total rooms
- Log-transformed sqft
- Target-encoded city (mean log-price per city in training split)
- Standardized numeric features

### ✔️ CNN Transfer Learning (EfficientNet)
- EfficientNetB0 and EfficientNetB3 pretrained on ImageNet
- Fine-tuned EfficientNetB0 (last 10 layers unfrozen)
- Strong augmentation pipeline (flip, zoom, rotation, translation, contrast)
- Extracted 128-dimensional visual embeddings representing house condition & style

### ✔️ Hybrid Architecture
The hybrid model concatenates:
```
[scaled tabular features] + [128-dim image embedding]
```
and feeds the combined vector into a tuned XGBoost regressor.

### ✔️ Performance

| Model | R² (test) |
|-------|-----------|
| Linear Regression (baseline) | ~0.40 |
| Random Forest | ~0.42 |
| XGBoost (baseline) | ~0.69 |
| XGBoost (feat-engineered + tuned) | ~0.78 |
| **Hybrid Model (Tabular + Images)** | **~0.80** |

*Results as reported in the original project.*

---

## 📁 Repository Structure
```
📦 Ironhack-FinalProject/
│
├── .gitignore                    # Git ignore rules
├── README.md                     # Project documentation
├── requirements.txt              # Pinned dependencies (Python 3.12, tested)
├── main.ipynb                    # Full training pipeline: tabular, CNN, hybrid
├── Presentation.pdf              # Project presentation slides
├── data/                         # Kaggle dataset goes here (not committed)
│
└── streamlit_app/
    ├── app.py                    # Streamlit demo application
    └── models/
        ├── city_target_enc.json          # Target-encoding mapping for city feature
        ├── config.json                   # Model configuration (image size, features)
        ├── effnetb0_tl_best.keras        # EfficientNetB0, frozen backbone (transfer learning)
        ├── effnetb0_tuned_last10.keras   # EfficientNetB0, last 10 layers fine-tuned (used for embeddings)
        ├── effnetb3_tl_best.keras        # EfficientNetB3, frozen backbone (experimental)
        ├── hybrid_features.npz           # Saved hybrid train/test feature matrices
        ├── hybrid_xgb_model.pkl          # Hybrid XGBoost model (tabular + image features)
        └── xgb_fe_tuned_pipeline.pkl     # Tabular-only preprocessing + tuned XGBoost
```

---

## 🧠 Modeling Pipeline

### 1️⃣ Exploratory Data Analysis
- Verified dataset integrity (15,474 rows, all images present)
- Visualized price distribution
- Modeled log-price due to heavy skew
- Analyzed correlations and feature relationships
- Identified location as a major driver → target encoding

### 2️⃣ Tabular Modeling

**Baseline models:** Linear Regression, Random Forest, XGBoost.

XGBoost clearly outperformed with R² ≈ 0.69, but not enough → needed better features.

**Feature engineering dramatically improved results:** spaciousness ratios, `log_sqft`, `total_rooms`, target encoding.

**XGBoost FE + tuning ⇒ R² ≈ 0.78**

### 3️⃣ Image Modeling (CNN)

**Models tested:**
- EfficientNetB0 (baseline)
- EfficientNetB3
- EfficientNetB0 fine-tuned (last 10 layers unfrozen)

CNNs alone performed poorly for price prediction, but the **fine-tuned EfficientNetB0** produced the best embeddings, used in the hybrid model.

**Images capture:** renovation quality, architectural style, curb appeal, landscaping, general exterior condition.

### 4️⃣ Hybrid Model

1. Encode tabular data → StandardScaler
2. Compute image embedding → fine-tuned EfficientNetB0
3. Concatenate → `[tabular_scaled | embedding_128]`
4. Predict log-price → XGBoost regressor
5. Convert back to USD using `expm1`

**Performance:** R² ≈ 0.80 on the test set. Visual features improved the model.

---

## 🚀 Streamlit Demo

A Streamlit app is included so users can:
- Input property details
- Optionally upload an exterior photo
- Choose a tabular-only or hybrid price estimate
- Get a final predicted price + explanation text

---

## ⚙️ Installation & Setup (Windows, Python 3.12)

### Prerequisites
- **Python 3.12** (64-bit). TensorFlow 2.19 does not support Python 3.13.
- Git
- Microsoft Visual C++ Redistributable (x64), required by TensorFlow on Windows

### 1️⃣ Clone the repository
```powershell
cd $HOME
git clone https://github.com/<your-username>/Ironhack-FinalProject.git
cd Ironhack-FinalProject
```

### 2️⃣ Create a virtual environment and install dependencies
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

<details>
<summary>Mac / Linux</summary>

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
</details>

### 3️⃣ Add the dataset
Download it from Kaggle and place `socal2.csv` and `socal_pics/` inside `data/` (see [Dataset](#-dataset)).

### 4️⃣ Run the notebook
Open `main.ipynb` in VS Code or Jupyter (`jupyter notebook`) and select the `.venv` interpreter as the kernel.
The setup cell should print `CSV found: True | images found: True`.

> **Note:** TensorFlow runs on the **CPU only** on native Windows, so training the three EfficientNet models (30 epochs each) can take several hours. For a quick run, lower `EPOCHS`. For GPU training, use WSL2 or Google Colab.

### 5️⃣ Run the Streamlit app
The trained models are already included, so no training is needed:
```powershell
cd streamlit_app
streamlit run app.py
```
The app downloads ImageNet weights on first run, so an internet connection is needed.

---

## 📦 Tested Versions

| Package | Version |
|---------|---------|
| Python | 3.12 |
| tensorflow | 2.19.0 |
| keras | 3.10.0 |
| scikit-learn | 1.6.1 |
| xgboost | 3.1.2 |
| numpy | 2.1.3 |
| pandas | 3.0.6 |
| dill | 0.4.1 |
| streamlit | 1.64.0 |


## 🧭 Future Work

- Incorporate multiple images per listing (interior + exterior)
- Use satellite imagery to capture neighborhood quality
- Explore multi-task learning (predict price + condition score)
- Implement SHAP for full model explainability
- Deploy the system as a full API + web app


