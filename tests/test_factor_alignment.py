import pandas as pd

from twse_factor_lab.validation.factor_alignment import validate_factor_matrix


def test_validate_factor_matrix_reports_alignment_and_missing_ratio():
    close_matrix = pd.DataFrame(
        {"1101": [10.0, 11.0], "1102": [20.0, 21.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    factor_matrix = pd.DataFrame(
        {"1101": [1.0, None], "1102": [2.0, 3.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    report = validate_factor_matrix(factor_matrix, close_matrix)

    assert report["is_aligned"] is True
    assert report["missing_ratio"] > 0


def test_validate_factor_matrix_detects_date_misalignment():
    close_matrix = pd.DataFrame({"1101": [10.0]}, index=pd.to_datetime(["2024-01-02"]))
    factor_matrix = pd.DataFrame({"1101": [1.0]}, index=pd.to_datetime(["2024-01-03"]))

    report = validate_factor_matrix(factor_matrix, close_matrix)

    assert report["is_aligned"] is False
