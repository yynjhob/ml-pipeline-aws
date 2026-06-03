import json
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model once at cold start
with open("outputs/model.pkl", "rb") as f:
    model = pickle.load(f)
logger.info("Model loaded at cold start")


def handler(event, context):
    try:
        # Parse request body
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event.get("body", event)

        text = body.get("text", "").strip()
        if not text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "text field is required"})
            }

        prediction = model.predict([text])[0]
        confidence = round(float(model.predict_proba([text]).max()), 4)
        sentiment  = "positive" if prediction == 1 else "negative"

        logger.info(f"Prediction: {sentiment} ({confidence})")

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "text":       text,
                "sentiment":  sentiment,
                "confidence": confidence
            })
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }