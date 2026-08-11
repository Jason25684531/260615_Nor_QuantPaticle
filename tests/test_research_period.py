import pytest

from twse_factor_lab.validation.research_period import validate_research_periods


def _periods():
    return {
        "in_sample": {"start": "2024-01-01", "end": "2024-01-10"},
        "out_of_sample": {"start": "2024-01-11", "end": "2024-01-20"},
    }


def test_validate_research_periods_accepts_ordered_periods():
    validate_research_periods(_periods(), "2024-01-01", "2024-01-31")


def test_research_period_uses_actual_ohlcv_start():
    validate_research_periods(_periods(), "2024-01-01", "2024-01-20")


def test_research_period_uses_actual_ohlcv_end():
    validate_research_periods(_periods(), "2024-01-01", "2024-01-20")


def test_oos_beyond_actual_data_end_rejected():
    periods = _periods()
    periods["out_of_sample"]["end"] = "2024-01-21"

    with pytest.raises(ValueError, match="outside"):
        validate_research_periods(periods, "2024-01-01", "2024-01-20")


def test_is_before_actual_data_start_rejected():
    periods = _periods()
    periods["in_sample"]["start"] = "2023-12-31"

    with pytest.raises(ValueError, match="outside"):
        validate_research_periods(periods, "2024-01-01", "2024-01-20")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("in_sample", "start", "2024-01-12"),
        ("out_of_sample", "start", "2024-01-21"),
        ("out_of_sample", "start", "2024-01-10"),
    ],
)
def test_validate_research_periods_rejects_invalid_order(section, key, value):
    periods = _periods()
    periods[section][key] = value

    with pytest.raises(ValueError):
        validate_research_periods(periods, "2024-01-01", "2024-01-31")


def test_validate_research_periods_rejects_outside_available_range():
    periods = _periods()
    periods["out_of_sample"]["end"] = "2024-02-01"

    with pytest.raises(ValueError, match="outside"):
        validate_research_periods(periods, "2024-01-01", "2024-01-31")
