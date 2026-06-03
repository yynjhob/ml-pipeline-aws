import pickle
import pytest
import os


MODEL_PATH = "outputs/model.pkl"


@pytest.fixture
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("model.pkl not found - run train.py first")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_model_loads(model):
    assert model is not None


def test_model_predicts(model):
    result = model.predict(["This is a great product"])
    assert result[0] in [0, 1]


def test_model_positive_sentiment(model):
    result = model.predict(["absolutely fantastic and wonderful"])
    assert result[0] == 1


def test_model_negative_sentiment(model):
    result = model.predict(["terrible and awful experience"])
    assert result[0] == 0


def test_pipeline_structure():
    """This test always runs - no model needed."""
    assert os.path.exists("src/train.py")
    assert os.path.exists("src/evaluate.py")
    assert os.path.exists("config.yaml")
    assert os.path.exists("requirements.txt")