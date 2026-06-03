import pickle
import pytest


def test_model_loads():
    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_model_predicts():
    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)
    result = model.predict(["This is a great product"])
    assert result[0] in [0, 1]


def test_model_positive_sentiment():
    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)
    result = model.predict(["absolutely fantastic and wonderful"])
    assert result[0] == 1


def test_model_negative_sentiment():
    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)
    result = model.predict(["terrible and awful experience"])
    assert result[0] == 0