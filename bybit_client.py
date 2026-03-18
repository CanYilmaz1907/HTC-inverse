import asyncio
from typing import Any, Dict, List, Optional

import aiohttp

from config import BybitConfig


class BybitClient:
    """
    Minimal async Bybit v5 REST client for market data.
    Uses only public endpoints; API key is optional and mainly reserved for future extensions.
    """

    def __init__(self, config: BybitConfig, session: Optional[aiohttp.ClientSession] = None) -> None:
        self._config = config
        self._session = session
        self._session_owner = False

    async def __aenter__(self) -> "BybitClient":
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._session_owner = True
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._session_owner and self._session is not None:
            await self._session.close()

    @property
    def base_url(self) -> str:
        return self._config.base_url

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        initial_backoff: float = 0.5,
    ) -> Dict[str, Any]:
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

        url = f"{self.base_url}{path}"
        attempt = 0
        backoff = initial_backoff

        while True:
            try:
                async with self._session.request(method, url, params=params, timeout=15) as resp:
                    if resp.status == 429:
                        # Rate limit – exponential backoff and retry
                        if attempt >= max_retries:
                            raise RuntimeError("Bybit rate limit exceeded and max retries reached.")
                        await asyncio.sleep(backoff)
                        attempt += 1
                        backoff *= 2
                        continue

                    resp.raise_for_status()
                    data = await resp.json()
                    if data.get("retCode") != 0:
                        raise RuntimeError(f"Bybit API error {data.get('retCode')}: {data.get('retMsg')}")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= max_retries:
                    raise RuntimeError(f"Bybit request failed after retries: {exc}") from exc
                await asyncio.sleep(backoff)
                attempt += 1
                backoff *= 2

    async def get_instruments_info(self) -> List[Dict[str, Any]]:
        """
        GET /v5/market/instruments-info?category=linear
        Returns list of linear instruments (USDT/USDC perpetuals and futures).
        """
        data = await self._request(
            "GET",
            "/v5/market/instruments-info",
            params={"category": self._config.category},
        )
        return data.get("result", {}).get("list", []) or []

    async def get_tickers(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        GET /v5/market/tickers?category=linear[&symbol=...]
        Returns tickers for one or all symbols.
        """
        params: Dict[str, Any] = {"category": self._config.category}
        if symbols and len(symbols) == 1:
            params["symbol"] = symbols[0]

        data = await self._request("GET", "/v5/market/tickers", params=params)
        items = data.get("result", {}).get("list", []) or []

        if symbols and len(symbols) > 1:
            # When requesting all tickers, filter client-side
            symbol_set = set(symbols)
            items = [it for it in items if it.get("symbol") in symbol_set]

        return items

    async def get_funding_history(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        GET /v5/market/funding/history?category=linear&symbol=...
        Mainly kept for potential extensions; not strictly required for current scan logic,
        since current fundingRate comes from the tickers endpoint.
        """
        data = await self._request(
            "GET",
            "/v5/market/funding/history",
            params={"category": self._config.category, "symbol": symbol, "limit": limit},
        )
        return data.get("result", {}).get("list", []) or []

    async def get_kline(
        self,
        symbol: str,
        interval: str = "5",
        limit: int = 1,
    ) -> List[List[Any]]:
        """
        GET /v5/market/kline?category=linear&symbol=...&interval=...
        Returns kline list where each item is:
        [startTime, open, high, low, close, volume, turnover]
        """
        data = await self._request(
            "GET",
            "/v5/market/kline",
            params={
                "category": self._config.category,
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            },
        )
        return data.get("result", {}).get("list", []) or []

