import pandas as pd
from src.utils import normalize_by_group_sum

def test_normalize_by_group_sum_basic():
    df = pd.DataFrame({
        "team":["A","A","A","B","B"],
        "week":[1,1,1,2,2],
        "val":[2.0, 3.0, 5.0, 0.0, 0.0]
    })
    out = normalize_by_group_sum(df, ["team","week"], "val", "norm")
    a = out[out["team"]=="A"]["norm"].sum()
    b = out[out["team"]=="B"]["norm"].sum()
    assert abs(a - 1.0) < 1e-9
    assert abs(b - 1.0) < 1e-9
