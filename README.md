# Titanic Survival Prediction - Docker Demo

A complete machine learning demo that trains a Random Forest classifier on the Titanic dataset and deploys it as a REST API using FastAPI and Docker.

## 🎯 Project Overview

This project demonstrates:
1. **Training**: Downloads the Titanic dataset from Kaggle and trains a machine learning model
2. **Deployment**: Serves the trained model via a FastAPI REST API
3. **Containerization**: Everything runs in Docker containers

## 📁 Project Structure

```
supervised-learning-docker-demo/
├── src/
│   ├── train.py          # Model training script
│   └── api.py            # FastAPI application
├── Dockerfile.train      # Docker image for training
├── Dockerfile.api        # Docker image for API
├── docker-compose.yml    # Orchestration configuration
├── requirements.txt      # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Internet connection (to download dataset from Kaggle)
- Kaggle API credentials (free Kaggle account required)

### Step 1: Set Up Kaggle API Credentials

1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Go to your account settings: https://www.kaggle.com/settings
3. Scroll down to the "API" section
4. Click "Create New API Token" - this downloads `kaggle.json`
5. Place the `kaggle.json` file in this project's root directory

**Important**: The `kaggle.json` file contains your credentials. Do NOT commit it to git (it's already in `.gitignore`).

**Note**: A `kaggle.json.template` file is provided as a reference for the expected format.

### Step 2: Run the Demo

Simply run:

```bash
docker compose up
```

This single command will:
1. Build both Docker images (training and API)
2. Download the Titanic dataset from Kaggle
3. Train the Random Forest model
4. Save the model as a pickle file
5. Start the FastAPI server with the trained model

The API will be available at `http://localhost:8000`

## 📊 Model Details

### Features Used
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Sex**: Gender (male/female)
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Passenger fare

### Target Variable
- **Survived**: 0 = Did not survive, 1 = Survived

### Algorithm
- Random Forest Classifier with 100 estimators

## 🔌 API Usage

### Interactive Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

### Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 3,
    "sex": "male",
    "age": 22.0,
    "sibsp": 1,
    "parch": 0,
    "fare": 7.25
  }'
```

Example response:
```json
{
  "survived": 0,
  "probability": 0.23,
  "prediction_text": "Did not survive"
}
```

#### 3. Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "pclass": 1,
      "sex": "female",
      "age": 28.0,
      "sibsp": 0,
      "parch": 0,
      "fare": 100.0
    },
    {
      "pclass": 3,
      "sex": "male",
      "age": 22.0,
      "sibsp": 1,
      "parch": 0,
      "fare": 7.25
    }
  ]'
```

## 🧪 Example Predictions

### High Survival Probability
First-class female passenger:
```json
{
  "pclass": 1,
  "sex": "female",
  "age": 28,
  "sibsp": 0,
  "parch": 0,
  "fare": 100.0
}
```

### Low Survival Probability
Third-class male passenger:
```json
{
  "pclass": 3,
  "sex": "male",
  "age": 22,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25
}
```

## 🛠️ Development

### View Training Logs
```bash
docker compose logs train
```

### View API Logs
```bash
docker compose logs api
```

### Rebuild Everything
```bash
docker compose down -v
docker compose up --build
```

### Stop Services
```bash
docker compose down
```

## 📚 Teaching Notes

This demo is ideal for teaching:

1. **Machine Learning Pipeline**
   - Data loading and preprocessing
   - Feature engineering (encoding categorical variables)
   - Model training and evaluation
   - Model serialization (pickle)

2. **API Development**
   - RESTful API design
   - Input validation with Pydantic
   - Error handling
   - API documentation with FastAPI

3. **DevOps & Docker**
   - Multi-stage containerization
   - Service orchestration with Docker Compose
   - Volume management for model persistence
   - Service dependencies

4. **Data Science Best Practices**
   - Handling missing values
   - Train-test split
   - Model evaluation metrics
   - Feature importance analysis

## 🔍 Understanding the Workflow

1. **Training Container** (`train` service):
   - Downloads Titanic dataset from Kaggle using `kagglehub`
   - Preprocesses data (handles missing values, encodes categorical features)
   - Trains Random Forest model
   - Saves model and label encoder to shared volume

2. **API Container** (`api` service):
   - Waits for training to complete
   - Loads trained model from shared volume
   - Starts FastAPI server
   - Serves predictions via REST API

3. **Shared Volume**:
   - Docker volume `models` persists trained model
   - Shared between training and API containers

## 📦 Dataset Source

The Titanic dataset is automatically downloaded from Kaggle:
- **Source**: https://www.kaggle.com/datasets/yasserh/titanic-dataset
- **Method**: Official Kaggle API Python package
- **Authentication**: Requires free Kaggle account and API token

## 💡 Tips for Students

1. Try modifying the model parameters in `src/train.py`
2. Add new features to improve accuracy
3. Experiment with different algorithms (SVM, Logistic Regression, etc.)
4. Add more API endpoints (e.g., model metrics, feature importance)
5. Implement model versioning
6. Add data validation and error handling

## 🐛 Troubleshooting

**Issue**: `FileNotFoundError: kaggle.json`
- **Solution**: Make sure you've downloaded your `kaggle.json` file from Kaggle and placed it in the project root directory. See "Step 1: Set Up Kaggle API Credentials" above.

**Issue**: API returns 503 "Model not loaded"
- **Solution**: Ensure training completed successfully. Check logs with `docker compose logs train`

**Issue**: Dataset download fails or "401 Unauthorized"
- **Solution**: Verify your `kaggle.json` file is valid. Try re-downloading it from https://www.kaggle.com/settings

**Issue**: Port 8000 already in use
- **Solution**: Stop other services using port 8000 or modify the port in `docker-compose.yml`

## 📝 License

This is an educational demo for teaching purposes.

## 🙏 Acknowledgments

- Titanic dataset from Kaggle
- FastAPI framework
- scikit-learn library

