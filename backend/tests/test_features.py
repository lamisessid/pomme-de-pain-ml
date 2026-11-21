import numpy as np
import pandas as pd

from app.ml.features import DYNAMIC_FEATURES, build_frame, risk_class


def _frame(sales):
    n = len(sales)
    dates = pd.date_range("2023-01-01", periods=n)
    ctx = pd.DataFrame({"temperature": 10.0, "rain": 0.0, "is_holiday": 0, "has_event": 0, "event_impact": 1.0}, index=dates)
    return build_frame(np.array(sales, dtype=np.float32), np.zeros(n, dtype=np.float32), dates, ctx, 10.0, np.zeros(2, dtype=np.float32))


def test_target_day_sales_do_not_leak():
    """Changer les ventes du dernier jour ne doit pas changer les features de ce jour."""
    base = list(range(1, 41))
    a = _frame(base)
    b = _frame(base[:-1] + [999])
    np.testing.assert_array_equal(a[-1], b[-1])


def test_lag_features():
    f = _frame([float(i) for i in range(1, 41)])
    i = DYNAMIC_FEATURES.index
    row = f[-1]
    assert np.isclose(row[i("lag1_sales")], 3.9)
    assert np.isclose(row[i("lag7_sales")], 3.3)
    assert np.isclose(row[i("mean7_sales")], np.mean(range(33, 40)) / 10)


def test_risk_class_thresholds():
    assert risk_class(np.array([100, 100, 100]), np.array([2, 8, 20])).tolist() == [0, 1, 2]
