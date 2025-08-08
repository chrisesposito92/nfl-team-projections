import numpy as np
from src.utils import allocate_total_by_weights

def test_allocate_total_by_weights_sums():
    total = 10.0
    weights = np.array([0.2, 0.3, 0.5])
    out = allocate_total_by_weights(total, weights)
    assert abs(out.sum() - total) < 1e-9

def test_allocate_total_by_weights_zero_weights_even_split():
    total = 9.0
    weights = np.array([0.0, 0.0, 0.0])
    out = allocate_total_by_weights(total, weights)
    assert np.allclose(out, np.array([3.0,3.0,3.0]))
