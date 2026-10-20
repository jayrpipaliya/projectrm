"""
API Client for Massive / Polygon Market Data.
Exclusively queries live API endpoints for Stocks, Options, and Reference Data.
No mock/fake data generators.
"""

import datetime
import requests
from typing import Dict, Any, Optional
import config


class MassiveApiClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key if api_key is not None else config.API_KEY
        self.base_url = (base_url or config.BASE_URL).rstrip("/")
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "ProjectRM-QuantEngine/1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a GET request to Polygon/Massive API with robust error handling."""
        if not self.api_key:
            return {
                "status": "ERROR",
                "error": "No API key configured in .env. Please set MASSIVE_API_KEY.",
                "status_code": 401
            }

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        query_params = dict(params) if params else {}
        if "apiKey" not in query_params and self.api_key:
            query_params["apiKey"] = self.api_key

        try:
            resp = self.session.get(url, headers=self._get_headers(), params=query_params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                data["_source"] = "LIVE_API"
                return data
            elif resp.status_code == 429:
                return {
                    "status": "RATE_LIMIT_EXCEEDED",
                    "error": "Polygon/Massive Free plan limit (5 requests/min) reached. Please wait 60s or query cached local SQL warehouse.",
                    "status_code": 429
                }
            elif resp.status_code in (401, 403):
                return {
                    "status": "UNAUTHORIZED",
                    "error": f"API error ({resp.status_code}): {resp.text}",
                    "status_code": resp.status_code
                }
            else:
                return {
                    "status": "ERROR",
                    "error": f"HTTP {resp.status_code}: {resp.text}",
                    "status_code": resp.status_code
                }
        except Exception as e:
            return {
                "status": "REQUEST_FAILED",
                "error": str(e)
            }

    # =========================================================================
    # STOCKS ENDPOINTS
    # =========================================================================

    def get_all_tickers(self, market: str = "stocks", limit: int = 20, search: Optional[str] = None) -> Dict[str, Any]:
        """Endpoint: /v3/reference/tickers"""
        params: Dict[str, Any] = {"market": market, "limit": limit, "active": "true"}
        if search:
            params["search"] = search
        return self._request("/v3/reference/tickers", params=params)

    def get_ticker_overview(self, ticker: str) -> Dict[str, Any]:
        """Endpoint: /v3/reference/tickers/{ticker}"""
        return self._request(f"/v3/reference/tickers/{ticker.upper()}")

    def get_ticker_types(self) -> Dict[str, Any]:
        """Endpoint: /v3/reference/tickers/types"""
        return self._request("/v3/reference/tickers/types")

    def get_related_tickers(self, ticker: str) -> Dict[str, Any]:
        """Endpoint: /v1/related-companies/{ticker}"""
        return self._request(f"/v1/related-companies/{ticker.upper()}")

    def get_stock_aggs(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = "day",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        adjusted: bool = True
    ) -> Dict[str, Any]:
        """Endpoint: /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}"""
        today = datetime.date.today()
        if not to_date:
            to_date = today.strftime("%Y-%m-%d")
        if not from_date:
            from_date = (today - datetime.timedelta(days=365)).strftime("%Y-%m-%d")

        endpoint = f"/v2/aggs/ticker/{ticker.upper()}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": "true" if adjusted else "false", "sort": "asc", "limit": 5000}
        return self._request(endpoint, params=params)

    def get_daily_market_summary(self, date_str: str) -> Dict[str, Any]:
        """Endpoint: /v2/aggs/grouped/locale/us/market/stocks/{date}"""
        return self._request(f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}")

    def get_daily_ticker_summary(self, ticker: str, date_str: str) -> Dict[str, Any]:
        """Endpoint: /v1/open-close/{stocksTicker}/{date}"""
        return self._request(f"/v1/open-close/{ticker.upper()}/{date_str}")

    def get_previous_close(self, ticker: str) -> Dict[str, Any]:
        """Endpoint: /v2/aggs/ticker/{stocksTicker}/prev"""
        return self._request(f"/v2/aggs/ticker/{ticker.upper()}/prev")

    def get_indicator_sma(self, ticker: str, timespan: str = "day", window: int = 50) -> Dict[str, Any]:
        """Endpoint: /v1/indicators/sma/{stockTicker}"""
        return self._request(f"/v1/indicators/sma/{ticker.upper()}", {"timespan": timespan, "window": window})

    def get_indicator_ema(self, ticker: str, timespan: str = "day", window: int = 20) -> Dict[str, Any]:
        """Endpoint: /v1/indicators/ema/{stockTicker}"""
        return self._request(f"/v1/indicators/ema/{ticker.upper()}", {"timespan": timespan, "window": window})

    def get_indicator_macd(self, ticker: str, timespan: str = "day") -> Dict[str, Any]:
        """Endpoint: /v1/indicators/macd/{stockTicker}"""
        return self._request(f"/v1/indicators/macd/{ticker.upper()}", {"timespan": timespan})

    def get_indicator_rsi(self, ticker: str, timespan: str = "day", window: int = 14) -> Dict[str, Any]:
        """Endpoint: /v1/indicators/rsi/{stockTicker}"""
        return self._request(f"/v1/indicators/rsi/{ticker.upper()}", {"timespan": timespan, "window": window})

    # =========================================================================
    # OPTIONS ENDPOINTS
    # =========================================================================

    def get_option_contracts(
        self,
        underlying_ticker: str = "QQQ",
        contract_type: Optional[str] = None,
        expiration_date: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Endpoint: /v3/reference/options/contracts"""
        params: Dict[str, Any] = {
            "underlying_ticker": underlying_ticker.upper(),
            "limit": limit
        }
        if contract_type and contract_type.upper() != "ALL":
            params["contract_type"] = contract_type.lower()
        if expiration_date:
            params["expiration_date"] = expiration_date

        return self._request("/v3/reference/options/contracts", params=params)

    def get_option_contract_overview(self, option_ticker: str) -> Dict[str, Any]:
        """Endpoint: /v3/reference/options/contracts/{options_ticker}"""
        return self._request(f"/v3/reference/options/contracts/{option_ticker}")

    def get_option_aggs(
        self,
        option_ticker: str,
        multiplier: int = 1,
        timespan: str = "day",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Endpoint: /v2/aggs/ticker/{optionsTicker}/range/{multiplier}/{timespan}/{from}/{to}"""
        today = datetime.date.today()
        if not to_date:
            to_date = today.strftime("%Y-%m-%d")
        if not from_date:
            from_date = (today - datetime.timedelta(days=90)).strftime("%Y-%m-%d")

        endpoint = f"/v2/aggs/ticker/{option_ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        return self._request(endpoint, params={"limit": 5000, "sort": "asc", "adjusted": "true"})

    def get_option_daily_ticker_summary(self, option_ticker: str, date_str: str) -> Dict[str, Any]:
        """Endpoint: /v1/open-close/{optionsTicker}/{date}"""
        return self._request(f"/v1/open-close/{option_ticker}/{date_str}")

    def get_option_previous_close(self, option_ticker: str) -> Dict[str, Any]:
        """Endpoint: /v2/aggs/ticker/{optionsTicker}/prev"""
        return self._request(f"/v2/aggs/ticker/{option_ticker}/prev")

    def get_market_status(self) -> Dict[str, Any]:
        """Endpoint: /v1/marketstatus/now"""
        return self._request("/v1/marketstatus/now")

    def get_market_holidays(self) -> Dict[str, Any]:
        """Endpoint: /v1/marketstatus/upcoming"""
        return self._request("/v1/marketstatus/upcoming")

    def get_exchanges(self, asset_class: str = "options") -> Dict[str, Any]:
        """Endpoint: /v3/reference/exchanges"""
        return self._request("/v3/reference/exchanges", {"asset_class": asset_class})

    def get_condition_codes(self, asset_class: str = "options") -> Dict[str, Any]:
        """Endpoint: /v3/reference/conditions"""
        return self._request("/v3/reference/conditions", {"asset_class": asset_class})

    def get_earnings_calendar(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieves historical and upcoming corporate earnings release dates.
        Resolves CIK via Polygon ticker overview / SEC mapping and queries official
        SEC EDGAR submission data for Form 8-K Item 2.02 (Results of Operations and Financial Condition).
        """
        import urllib.request
        import json

        ticker = ticker.upper().strip()

        # Step 1: Attempt to get CIK from Polygon ticker overview
        cik = None
        try:
            overview = self.get_ticker_overview(ticker)
            if overview.get("status") == "OK" and overview.get("results", {}).get("cik"):
                cik = str(overview["results"]["cik"]).zfill(10)
        except Exception:
            pass

        # Step 2: Fallback to SEC company_tickers.json mapping if Polygon overview lacks CIK
        if not cik:
            try:
                mapping_url = "https://www.sec.gov/files/company_tickers.json"
                req = urllib.request.Request(mapping_url, headers={
                    "User-Agent": "ProjectRM Research/1.0 (quant@projectrm.internal)"
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    for entry in data.values():
                        if entry.get("ticker", "").upper() == ticker:
                            cik = str(entry.get("cik_str", "")).zfill(10)
                            break
            except Exception:
                pass

        if not cik:
            return {
                "status": "ERROR",
                "ticker": ticker,
                "error": f"Could not resolve CIK for ticker {ticker}",
                "results": []
            }

        # Step 3: Fetch submissions from SEC EDGAR
        try:
            sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            req = urllib.request.Request(sub_url, headers={
                "User-Agent": "ProjectRM Research/1.0 (quant@projectrm.internal)"
            })
            with urllib.request.urlopen(req, timeout=12) as resp:
                sub_data = json.loads(resp.read().decode("utf-8"))

            recent = sub_data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            filing_dates = recent.get("filingDate", [])
            items_list = recent.get("items", [])
            accept_times = recent.get("acceptanceDateTime", [])

            results = []
            seen_dates = set()
            for i, form in enumerate(forms):
                items = str(items_list[i]) if i < len(items_list) else ""
                fdate = filing_dates[i] if i < len(filing_dates) else ""
                # Item 2.02 is standard for "Results of Operations and Financial Condition" (Earnings release)
                if form == "8-K" and "2.02" in items and fdate and fdate not in seen_dates:
                    seen_dates.add(fdate)
                    acc_time = accept_times[i] if i < len(accept_times) else ""
                    results.append({
                        "ticker": ticker,
                        "earnings_date": fdate,
                        "acceptance_time": acc_time,
                        "form": "8-K",
                        "items": items,
                        "source": "SEC_EDGAR_8K_ITEM_2.02"
                    })

            return {
                "status": "OK",
                "ticker": ticker,
                "cik": cik,
                "count": len(results),
                "results": results
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "ticker": ticker,
                "error": str(e),
                "results": []
            }
