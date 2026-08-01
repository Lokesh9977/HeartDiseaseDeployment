# Heart Disease Prediction — End-to-End ML Deployment

A machine learning model that predicts whether a patient is at risk of heart
disease based on clinical parameters, served via a Flask REST API and
deployed on Render.

## Project Structure

```
HeartDiseaseDeployment/
│
├── app.py                 # Flask REST API
├── train_model.py         # Data preprocessing + model training script
├── model.pkl              # Trained model (RandomForestClassifier)
├── feature_names.pkl      # Feature order used by the model
├── heart.csv              # Heart Disease dataset
├── requirements.txt       # Python dependencies
├── Procfile                # Render/Gunicorn start command
├── README.md
├── templates/
│   └── index.html         # Simple API info page
└── static/
```

## Dataset

Heart Disease Prediction Dataset (303 records, 13 clinical features, binary
target: 1 = heart disease present, 0 = absent).

Columns: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
oldpeak, slope, ca, thal, target`

## 1. Data Understanding & Preprocessing

- Loaded with Pandas, first 5 records inspected.
- Numerical features: all 13 clinical columns above.
- Target variable: `target`.
- No missing values in the dataset.
- Split: 80% train / 20% test (stratified).

## 2. Model Development

- Algorithm: **Random Forest Classifier** (`n_estimators=200`).
- Evaluation metric: **Accuracy Score** → **~0.82** on the held-out test set.
- Model saved with `joblib` as `model.pkl` (feature order saved as
  `feature_names.pkl`).

Retrain locally with:

```powershell
python train_model.py
```

## 3. API Development (Flask)

| Endpoint    | Method | Description                          |
|-------------|--------|---------------------------------------|
| `/`         | GET    | API info page                        |
| `/health`   | GET    | Health check                         |
| `/predict`  | POST   | Accepts patient JSON, returns prediction |

### Example Request

```json
POST /predict
{
  "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
  "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
  "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}
```

### Example Response

```json
{
  "prediction": "Heart Disease Detected",
  "probability": 0.655
}
```

## Run Locally (Windows / PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

The API will be available at `http://127.0.0.1:5000`.

## Deployment on Render

1. Push this repository to a **public GitHub repo**.
2. Go to [render.com](https://render.com) → **New +** → **Web Service**.
3. Connect your GitHub repo.
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3
5. Deploy. Render will assign a public URL such as
   `https://heartdiseasedeployment.onrender.com`.

### Live Deployment URL

**Render URL:** `https://heartdiseasedeployment-9itc.onrender.com`

## Conclusion

The Random Forest model achieved an accuracy of approximately 82% on the
held-out test set, showing reasonably strong performance for a small
clinical dataset with 13 features. Preprocessing was straightforward since
the dataset had no missing values, though care was needed to keep feature
order consistent between training and inference. The main challenges
during deployment were making sure the exact scikit-learn/pandas versions
matched between the training environment and the Render runtime, handling
malformed or missing JSON fields gracefully in the API, and configuring
Render's build/start commands correctly so the app boots with Gunicorn
instead of the Flask development server. This project highlights why
MLOps practices matter: version-controlled code, reproducible
environments (`requirements.txt`), model serialization, and automated
cloud deployment together turn a one-off notebook experiment into a
reliable, reusable service that other systems can call in production.

## Learning Outcomes Covered

- Built and evaluated a machine learning classification model.
- Saved and loaded a trained model using Joblib.
- Developed a REST API using Flask.
- Managed project code using GitHub.
- Prepared the app for cloud deployment on Render.
- Understood MLOps fundamentals: packaging, version control, deployment,
  and serving predictions via an API.
