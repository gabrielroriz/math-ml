import numpy as np
from src.nn import log_loss

def test_log_loss_perfect_1():
    y = 1
    y_hat = 0.999999
    assert np.isclose(log_loss(y, y_hat), -np.log(y_hat), atol=1e-5)

def test_log_loss_perfect_0():
    y = 0
    y_hat = 0.000001
    assert np.isclose(log_loss(y, y_hat), -np.log(1 - y_hat), atol=1e-5)

def test_log_loss_half_probability():
    y = 1
    y_hat = 0.5
    assert np.isclose(log_loss(y, y_hat), -np.log(0.5), atol=1e-5)

def test_log_loss_typical_case():
    y = 0
    y_hat = 0.3
    expected = -np.log(1 - y_hat)
    assert np.isclose(log_loss(y, y_hat), expected, atol=1e-5)

def test_log_loss_average():
    y = np.array([1, 0, 1])
    y_hat = np.array([0.9, 0.2, 0.8])
    result = np.mean([log_loss(yi, yhi) for yi, yhi in zip(y, y_hat)])
    expected = np.mean([
        -np.log(0.9),
        -np.log(0.8),
        -np.log(0.8)
    ])
    assert np.isclose(result, expected, atol=1e-5)