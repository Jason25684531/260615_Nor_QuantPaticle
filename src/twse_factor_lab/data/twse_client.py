"""TWSE OpenAPI client."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from twse_factor_lab.data.endpoints import get_endpoint


class TWSEClientError(RuntimeError):
    """Raised when TWSE data cannot be fetched or parsed."""


class TWSEClient:
    """Small TWSE OpenAPI client that returns raw DataFrames."""

    def __init__(
        self,
        base_url: str = "https://openapi.twse.com.tw/v1",
        timeout: int = 30,
        session: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def fetch_dataframe(self, endpoint_key: str) -> pd.DataFrame:
        """Fetch a registered endpoint and return the raw JSON array as a DataFrame."""
        try:
            endpoint = get_endpoint(endpoint_key)
        except KeyError as exc:
            raise TWSEClientError(f"Failed to resolve endpoint {endpoint_key}") from exc

        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise TWSEClientError(
                f"Failed to fetch endpoint {endpoint_key}: {url}"
            ) from exc

        if not isinstance(payload, list):
            raise TWSEClientError(
                f"Endpoint {endpoint_key} returned invalid payload type: "
                f"{type(payload).__name__}"
            )
        if not payload:
            raise TWSEClientError(f"Endpoint {endpoint_key} returned an empty payload")

        return pd.DataFrame(payload)
