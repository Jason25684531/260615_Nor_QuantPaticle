import pandas as pd
import pytest

from twse_factor_lab.data.twse_client import TWSEClient, TWSEClientError


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return self.response


def test_twse_client_initializes_with_base_url_timeout_and_session():
    session = FakeSession(FakeResponse([]))
    client = TWSEClient(
        base_url="https://openapi.twse.com.tw/v1/",
        timeout=7,
        session=session,
    )

    assert client.base_url == "https://openapi.twse.com.tw/v1"
    assert client.timeout == 7
    assert client.session is session


def test_twse_client_fetches_raw_dataframe_from_endpoint():
    session = FakeSession(FakeResponse([{"Code": "2330", "Name": "TSMC"}]))
    client = TWSEClient(session=session)

    frame = client.fetch_dataframe("listed_companies")

    assert isinstance(frame, pd.DataFrame)
    assert frame.to_dict("records") == [{"Code": "2330", "Name": "TSMC"}]
    assert session.calls == [("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", 30)]


def test_twse_client_reports_request_failure_with_endpoint_name():
    session = FakeSession(FakeResponse([], status_error=RuntimeError("boom")))
    client = TWSEClient(session=session)

    with pytest.raises(TWSEClientError, match="listed_companies"):
        client.fetch_dataframe("listed_companies")


def test_twse_client_rejects_invalid_payload_with_endpoint_name():
    session = FakeSession(FakeResponse({"not": "a list"}))
    client = TWSEClient(session=session)

    with pytest.raises(TWSEClientError, match="listed_companies"):
        client.fetch_dataframe("listed_companies")


def test_twse_client_rejects_empty_payload_with_endpoint_name():
    session = FakeSession(FakeResponse([]))
    client = TWSEClient(session=session)

    with pytest.raises(TWSEClientError, match="listed_companies"):
        client.fetch_dataframe("listed_companies")
