import numpy as np
from src.nn import sigmoid

def test_sigmoid_zero():
    assert np.isclose(sigmoid(0), 0.5)

def test_sigmoid_positive():
    assert sigmoid(2) > 0.5

def test_sigmoid_negative():
    assert sigmoid(-2) < 0.5

def test_sigmoid_large_positive():
    assert np.isclose(sigmoid(100), 1.0, atol=1e-5)

def test_sigmoid_large_negative():
    assert np.isclose(sigmoid(-100), 0.0, atol=1e-5)
