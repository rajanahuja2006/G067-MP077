import pandas as pd

from mp077.data_checks import (check_duplicates, check_spelling_variants,
                               check_unit_mixup)


def _df():
    return pd.DataFrame({
        "locality": ["A"] * 3 + ["B"] * 3 + ["C"] * 4,
        "discount_pct": [10, 12, 14, 11, 13, 12, 0.1, 0.12, 0.14, 0.11],
        "flu_cases_reported": [5] * 10, "temp_c": [5] * 10,
        "nearby_clinics": [5] * 10, "stock_lines": [5] * 10,
        "distance_to_hospital_km": [5] * 10,
        "shop_type": ["chain", "Chain", "CHAIN", "ind.", "independent"] * 2,
    })


def test_duplicates_counts_extra_copies():
    d = pd.concat([_df(), _df().iloc[:2]])
    assert check_duplicates(d)["count"] == 2


def test_unit_mixup_finds_group_and_column():
    r = check_unit_mixup(_df())
    assert r["present"] and r["column"] == "discount_pct" and r["group_value"] == "C"


def test_spelling_variants_detected():
    clean = pd.DataFrame({"shop_type": ["chain", "independent", "hospital_store"]})
    r = check_spelling_variants(_df(), clean)
    assert r["present"] and r["true_levels"] == 3
