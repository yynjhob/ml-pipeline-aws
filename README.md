[![ML Pipeline](https://github.com/yynjhob/ml-pipeline-aws/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/yynjhob/ml-pipeline-aws/actions/workflows/ml_pipeline.yml)
 
 Automated
ML
training
and
deployment
pipeline
using
MLflow
Docker
and
GitHub
Actions.

# ML Pipeline on AWS
![ML Pipeline](YOUR-BADGE-URL)

An end-to-end MLOps pipeline that automatically trains, evaluates, and deploys a sentiment classification model using GitHub Actions, Docker, and AWS.

---

## Architecture

```mermaid
flowchart LR
    A[Push to GitHub] --> B[GitHub Actions]
    B --> C[Run Tests]
    C --> D[Train Model]
    D --> E[Evaluate + Promotion Gate]
    E --> F[Build Docker Image]
    F --> G[Push to Amazon ECR]
    G --> H[Deploy to EC2]
    H --> I[Live API Endpoint]
```

---

## What this pipeline does

On every push to `main` the pipeline automatically:

1. Runs unit tests against the model and repo structure
2. Trains a sentiment classifier on the SST-2 dataset stored in S3
3. Evaluates the model - only promotes if accuracy exceeds 80%
4. Builds a Docker image and pushes to Amazon ECR
5. Serves predictions via a FastAPI REST endpoint on EC2

---

## Model performance

| Metric | Score |
|---|---|
| Accuracy | 0.8062 |
| F1 | 0.8169 |
| Precision | 0.7871 |
| Recall | 0.8491 |

---

## Tech stack

| Layer | Technology |
|---|---|
| Model | Scikit-learn TF-IDF + Logistic Regression |
| Experiment tracking | MLflow |
| Containerization | Docker |
| Image registry | Amazon ECR |
| Compute | Amazon EC2 |
| Storage | Amazon S3 |
| CI/CD | GitHub Actions |
| API | FastAPI + Uvicorn |
| Drift detection | Scipy KS test |

---

## Project structure
```
ml-pipeline-aws/
├── src/
│   ├── train.py              # Training pipeline with MLflow tracking
│   ├── evaluate.py           # Evaluation with promotion gate and drift detection
│   ├── app.py                # FastAPI serving endpoint
│   ├── monitor.py            # CloudWatch metrics and model health checks
│   └── lambda_handler.py     # AWS Lambda handler
├── tests/
│   └── test_model.py         # Unit tests
├── .github/workflows/
│   └── ml_pipeline.yml       # CI/CD pipeline
├── Dockerfile
├── requirements.txt
├── requirements-serve.txt
├── config.yaml
└── RUNBOOK.md
```
---

## API usage

**Health check:**
```bash
curl http://YOUR-EC2-IP:8000/health
```

**Predict sentiment:**
```bash
curl -X POST "http://YOUR-EC2-IP:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing"}'
```

**Response:**
```json
{
  "text": "This product is absolutely amazing",
  "sentiment": "positive",
  "confidence": 0.89
}
```

---

## Setup

**Prerequisites:** Python 3.11, Docker, AWS CLI configured

```bash
# Clone and install
git clone https://github.com/YOUR-USERNAME/ml-pipeline-aws.git
cd ml-pipeline-aws
pip install -r requirements.txt

# Train locally
python src/train.py

# Evaluate
python src/evaluate.py

# Run API locally
uvicorn src.app:app --reload
```

---

## Future Work

- **Replace EC2 with ECS or EKS** for auto-scaling and zero-downtime deployments
- **Replace TF-IDF with a fine-tuned transformer** (DistilBERT) for higher accuracy
- **Add A/B testing** to route traffic between model versions and measure business impact
- **Implement full observability** — structured logging, distributed tracing, and alerting on prediction drift
---

## Author

Built by yynjohb as a portfolio project demonstrating end-to-end MLOps on AWS.
