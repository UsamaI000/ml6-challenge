# Sensor Data GMM Classification System

## Overview

This project implements a Semi-Supervised Sensor Data Classification System. It utilizes a Gaussian Mixture Model (GMM) to cluster sensor data and assigns class labels based on a limited set of labeled examples. The system includes a training pipeline, an inference engine with confidence scoring, and a FastAPI-based prediction service.

## Features

- **Semi-Supervised Learning**: Leverages both labeled and unlabeled data using GMM clustering.
- **Auto-Assignment Logic**: Assigns labels automatically only when predictions meet strict confidence, support, and purity criteria.
- **REST API**: Provides real-time prediction endpoints via FastAPI.
- **Comprehensive Evaluation**: Tracks accuracy, Adjusted Rand Index (ARI), and operational trade-offs.

## Project Structure

```
.
├── app/
│   ├── app.py           # FastAPI application entry point
│   ├── config.py        # Project configuration and hyperparameters
│   ├── inference.py     # Inference engine and prediction logic
│   ├── train.py         # Model training and evaluation script
│   ├── baseline_outputs/# Directory for trained models and artifacts
│   └── data/            # Data directory
├── notebooks/           # Jupyter notebooks for exploration
└── requirements.txt     # Python dependencies
```

## Setup & Installation

1. **Create a virtual environment**:

   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r app/requirements.txt
   ```

## Usage

### 1. Training the Model

To train the model, run the `train.py` script. This will process the data, train the GMM, and generate artifacts in the `app/baseline_outputs/` directory.

```bash
cd app
python train.py
```

### 2. Running the API

Start the FastAPI server using `uvicorn`:

```bash
cd app
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`. You can access the automatic documentation at `http://localhost:8000/docs`.

### 3. Making Predictions

Send a POST request to `/predict` with sensor data:

```json
POST /predict
{
  "features": {
    "Sensor 0": 0.12,
    "Sensor 1": 0.45,
    ...
  }
}
```

## Methodology

### Algorithm

The system uses a **Gaussian Mixture Model (GMM)** to model the density of the sensor data.

- **Model Selection**: The optimal number of clusters ($K$) is selected using the **Bayesian Information Criterion (BIC)**.
- **Label Mapping**: Each cluster is mapped to the majority class of the labeled data points within it.

### Confidence & Auto-assignment

Predictions are marked as `auto_assign=True` only if they pass strict reliability gates:

1.  **Confidence**: Posterior probability $\ge$ `CONF_THRESH` (default: 0.85).
2.  **Support**: Cluster has at least `MIN_SUPPORT` labeled examples (default: 4).
3.  **Purity**: Cluster's majority class accounts for $\ge$ `MIN_PURITY` of labeled data (default: 80%).
