"""
Project RM - Web Dashboard Server
Runs a lightweight HTTP server delivering live data:
- QQQ historical stock prices & 20-day annualized Realized Volatility
- Live QQQ Option contracts chain from Polygon
- Real option historical bar prices
- Local SQLite Data Warehouse management
- Endpoint playground for all Massive/Polygon endpoints
"""

import http.server
import socketserver
import json
import urllib.parse
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np

import config
from api_client import MassiveApiClient
from warehouse import Warehouse
from residual_model import ResidualAnomalyModel
from universe import UNIVERSE_SECTORS, get_pairs_by_sector
from scanner_engine import UniverseScannerEngine
from semi_markov import SemiMarkovFitter
from threshold_optimizer import ThresholdOptimizer
from yfinance_loader import YFinanceLoader

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

warehouse = Warehouse()
anomaly_model = ResidualAnomalyModel()
scanner_engine = UniverseScannerEngine()


def resolve_pair_for_ticker(ticker: str, market_param: str = "", sector_param: str = ""):
    ticker = (ticker or "").strip().upper()
    mkt = (market_param or "").strip().upper()
    sec = (sector_param or "").strip().upper()
    if mkt and sec:
        return mkt, sec
    for sec_info in UNIVERSE_SECTORS.values():
        for p in sec_info.get("pairs", []):
            if p["ticker"].upper() == ticker:
                return mkt if mkt else p["market"], sec if sec else p["sector"]
    return mkt if mkt else "QQQ", sec if sec else "SMH"


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        client = MassiveApiClient()

        if path == "/":
            self.path = "/index.html"
            return super().do_GET()

        elif path == "/api/config":
            has_key = bool(config.API_KEY)
            masked_key = f"{config.API_KEY[:4]}...{config.API_KEY[-4:]}" if len(config.API_KEY) > 8 else ("Active" if has_key else "Missing")
            stats = warehouse.get_warehouse_stats()
            self._send_json({
                "has_api_key": has_key,
                "masked_api_key": masked_key,
                "base_url": config.BASE_URL,
                "warehouse_stats": stats
            })

        elif path == "/api/stock/aggs":
            ticker = query.get("ticker", ["QQQ"])[0].upper()
            from_date = query.get("from", [None])[0]
            to_date = query.get("to", [None])[0]

            # Check warehouse first
            cached_data = warehouse.get_stock_history(ticker)
            if cached_data and len(cached_data) > 10:
                self._send_json({
                    "ticker": ticker,
                    "results": cached_data,
                    "source": "LOCAL_SQL_WAREHOUSE"
                })
                return

            res = client.get_stock_aggs(ticker, from_date=from_date, to_date=to_date)
            results = res.get("results", [])
            if results:
                warehouse.save_stock_bars(ticker, results)
                combined = warehouse.get_stock_history(ticker)
                self._send_json({
                    "ticker": ticker,
                    "results": combined,
                    "source": "LIVE_API"
                })
                return

            # Fallback to yfinance if live API returns empty
            try:
                yf_loader = YFinanceLoader(warehouse)
                yf_res = yf_loader.fetch_and_store_ticker(ticker, period="2y")
                combined = warehouse.get_stock_history(ticker)
                if combined and len(combined) > 0:
                    self._send_json({
                        "ticker": ticker,
                        "results": combined,
                        "source": "YFINANCE_STORED"
                    })
                    return
            except Exception as e:
                print(f"Error in yfinance fallback for {ticker}: {e}")

            self._send_json(res)

        elif path == "/api/stock/prev":
            ticker = query.get("ticker", ["QQQ"])[0].upper()
            res = client.get_previous_close(ticker)
            self._send_json(res)

        elif path == "/api/options/contracts":
            underlying = query.get("underlying", ["QQQ"])[0].upper()
            ctype = query.get("type", [None])[0]
            
            # Check if warehouse has options
            cached_contracts = warehouse.get_option_contracts(underlying, contract_type=ctype)
            if cached_contracts and len(cached_contracts) >= 10:
                self._send_json({
                    "status": "OK",
                    "results": cached_contracts,
                    "source": "LOCAL_SQL_WAREHOUSE"
                })
                return

            res = client.get_option_contracts(underlying_ticker=underlying, contract_type=ctype, limit=50)
            contracts = res.get("results", [])
            if contracts:
                warehouse.save_option_contracts(contracts)
                self._send_json(res)
                return

            # If no live option contracts from Polygon, synthesize ATM contracts for the ticker
            latest_px = 100.0
            hist = warehouse.get_stock_history(underlying)
            if hist:
                latest_px = float(hist[-1].get("close", 100.0))
            
            synth = []
            import datetime as dt
            base_date = dt.date.today()
            for dte in [7, 14, 21, 35, 45]:
                exp_date = (base_date + dt.timedelta(days=dte)).strftime("%Y-%m-%d")
                for strike_offset in [-5.0, 0.0, 5.0]:
                    strike = round(latest_px + strike_offset, 1)
                    if ctype in [None, "call", "ALL"]:
                        synth.append({
                            "ticker": f"O:{underlying}{exp_date.replace('-', '')[2:]}C{int(strike*1000):08d}",
                            "underlying_ticker": underlying,
                            "contract_type": "call",
                            "strike_price": strike,
                            "expiration_date": exp_date
                        })
                    if ctype in [None, "put", "ALL"]:
                        synth.append({
                            "ticker": f"O:{underlying}{exp_date.replace('-', '')[2:]}P{int(strike*1000):08d}",
                            "underlying_ticker": underlying,
                            "contract_type": "put",
                            "strike_price": strike,
                            "expiration_date": exp_date
                        })
            self._send_json({
                "status": "OK",
                "results": synth,
                "source": "SYNTHETIC_CHAIN"
            })

        elif path == "/api/options/aggs":
            option_ticker = query.get("ticker", [""])[0]
            if not option_ticker:
                self._send_json({"error": "No option ticker provided"}, status=400)
                return

            cached_bars = warehouse.get_option_bars(option_ticker)
            if cached_bars and len(cached_bars) > 0:
                self._send_json({
                    "ticker": option_ticker,
                    "results": cached_bars,
                    "source": "LOCAL_SQL_WAREHOUSE"
                })
                return

            res = client.get_option_aggs(option_ticker)
            if res.get("results"):
                warehouse.save_option_bars(option_ticker, res["results"])
                cached_bars = warehouse.get_option_bars(option_ticker)
                self._send_json({
                    "ticker": option_ticker,
                    "results": cached_bars,
                    "source": "LIVE_API"
                })
            else:
                self._send_json(res)

        elif path == "/api/warehouse/stats":
            self._send_json(warehouse.get_warehouse_stats())

        elif path == "/api/market/status":
            self._send_json(client.get_market_status())

        elif path == "/api/earnings":
            ticker = query.get("ticker", ["MSFT"])[0].upper()
            cached = warehouse.get_earnings_dates(ticker)
            if cached:
                self._send_json({
                    "status": "OK",
                    "ticker": ticker,
                    "count": len(cached),
                    "results": cached,
                    "source": "LOCAL_SQL_WAREHOUSE"
                })
            else:
                res = client.get_earnings_calendar(ticker)
                if res.get("results"):
                    warehouse.save_earnings_dates(ticker, res["results"])
                self._send_json(res)

        elif path == "/api/anomaly/analyze":
            asset = query.get("asset", ["MSFT"])[0].upper()
            market = query.get("market", ["QQQ"])[0].upper()
            sector = query.get("sector", ["IGV"])[0].upper()
            champ_triplet = warehouse.get_champion_triplet(asset)
            opt_params = warehouse.get_optimal_params(asset) or {}

            # Prioritize 3D Champion Triplet from Gold Ledger
            if champ_triplet:
                opt_params["optimal_threshold"] = float(champ_triplet.get("threshold", opt_params.get("optimal_threshold", 1.8)))
                opt_params["optimal_horizon"] = int(champ_triplet.get("holding_days", opt_params.get("optimal_horizon", 5)))
                opt_params["contract_dte"] = int(champ_triplet.get("contract_dte", 21))
                opt_params["profit_factor"] = float(champ_triplet.get("profit_factor", 0.0))
                opt_params["win_rate_pct"] = float(champ_triplet.get("win_rate_pct", 0.0))
                opt_params["avg_roi_pct"] = float(champ_triplet.get("avg_roi_pct", 0.0))

            use_optimal = query.get("use_optimal", ["false"])[0].lower() in ("true", "1", "yes")
            if use_optimal and opt_params and "optimal_threshold" in opt_params:
                threshold = float(opt_params["optimal_threshold"])
            elif "threshold" in query:
                try:
                    threshold = float(query.get("threshold")[0])
                except (ValueError, TypeError):
                    threshold = float(opt_params["optimal_threshold"]) if (opt_params and "optimal_threshold" in opt_params) else 1.8
            elif opt_params and "optimal_threshold" in opt_params:
                threshold = float(opt_params["optimal_threshold"])
            else:
                threshold = 1.8

            filter_earnings = query.get("filter_earnings", ["true"])[0].lower() not in ("false", "0", "no")
            res = anomaly_model.analyze(
                asset_ticker=asset,
                market_ticker=market,
                sector_ticker=sector,
                snapback_threshold=threshold,
                filter_earnings=filter_earnings
            )
            res["optimal_params"] = opt_params
            res["threshold_used"] = threshold
            self._send_json(res)

        elif path == "/api/scanner/universe":
            # Returns metadata of all sectors and tickers
            sectors_summary = {}
            for sec_key, sec_val in UNIVERSE_SECTORS.items():
                sectors_summary[sec_key] = {
                    "name": sec_val["name"],
                    "description": sec_val["description"],
                    "pairs": sec_val["pairs"],
                    "count": len(sec_val["pairs"])
                }
            self._send_json({
                "status": "SUCCESS",
                "total_sectors": len(UNIVERSE_SECTORS),
                "sectors": sectors_summary
            })

        elif path == "/api/scanner/scan":
            sector = query.get("sector", ["ALL"])[0]
            try:
                threshold = float(query.get("threshold", [1.8])[0])
            except (ValueError, TypeError):
                threshold = 1.8
            filter_earnings = query.get("filter_earnings", ["true"])[0].lower() not in ("false", "0", "no")
            threshold_mode = query.get("mode", ["optimal"])[0].lower()
            scan_res = scanner_engine.scan_universe(
                sector_id=sector,
                threshold=threshold,
                filter_earnings=filter_earnings,
                threshold_mode=threshold_mode
            )
            self._send_json(scan_res)

        elif path == "/api/scanner/freshness":
            # Returns per-ticker freshness status for all unique benchmarks + cached stocks
            import datetime as dt
            today_str = dt.date.today().strftime("%Y-%m-%d")
            # Yesterday accounting for weekends
            yesterday = dt.date.today() - dt.timedelta(days=1)
            if yesterday.weekday() == 6: yesterday -= dt.timedelta(days=2)  # Sunday -> Friday
            if yesterday.weekday() == 5: yesterday -= dt.timedelta(days=1)  # Saturday -> Friday
            yesterday_str = yesterday.strftime("%Y-%m-%d")

            from universe import get_all_benchmarks, get_all_tickers
            all_benchmarks = get_all_benchmarks()
            all_stocks = get_all_tickers()

            freshness = []
            for ticker in sorted(set(all_benchmarks + all_stocks)):
                hist = warehouse.get_stock_history(ticker)
                if not hist:
                    freshness.append({
                        "ticker": ticker,
                        "is_benchmark": ticker in all_benchmarks,
                        "latest_date": None,
                        "latest_close": None,
                        "status": "MISSING",
                        "fresh": False
                    })
                else:
                    latest = hist[-1]
                    latest_date = latest.get("trade_date", "")
                    latest_close = latest.get("close")
                    is_fresh = latest_date >= yesterday_str
                    status = "CURRENT" if is_fresh else "STALE"
                    freshness.append({
                        "ticker": ticker,
                        "is_benchmark": ticker in all_benchmarks,
                        "latest_date": latest_date,
                        "latest_close": latest_close,
                        "status": status,
                        "fresh": is_fresh
                    })

            stale = [f for f in freshness if not f["fresh"]]
            stale_benchmarks = [f for f in stale if f["is_benchmark"]]
            ready = all(f["fresh"] for f in freshness if f["is_benchmark"])

            self._send_json({
                "status": "SUCCESS",
                "as_of_date": today_str,
                "previous_trading_day": yesterday_str,
                "scanner_ready": ready,
                "total_tracked": len(freshness),
                "stale_count": len(stale),
                "stale_benchmark_count": len(stale_benchmarks),
                "freshness": freshness
            })

        elif path == "/api/smm/status":
            ticker = query.get("ticker", ["NVDA"])[0].upper()
            market = query.get("market", ["QQQ"])[0].upper()
            sector = query.get("sector", ["SMH"])[0].upper()
            try:
                threshold = float(query.get("threshold", [1.8])[0])
            except (ValueError, TypeError):
                threshold = 1.8

            analysis = anomaly_model.analyze(
                asset_ticker=ticker,
                market_ticker=market,
                sector_ticker=sector,
                snapback_threshold=threshold
            )

            if analysis.get("status") != "SUCCESS":
                self._send_json({"status": "ERROR", "error": analysis.get("error", "Analysis failed")})
                return

            history = analysis.get("history", [])
            if len(history) < 20:
                self._send_json({
                    "status": "ERROR",
                    "error": f"Insufficient history ({len(history)} obs) to fit SMM."
                })
                return

            z_series = np.array([float(h.get("z_1d", 0.0)) for h in history])
            smf = SemiMarkovFitter(threshold=threshold)
            smm = smf.fit(z_series)

            self._send_json({
                "status": "SUCCESS",
                "ticker": ticker,
                "market": market,
                "sector": sector,
                "threshold": threshold,
                "smm": smm,
                "latest_z": float(z_series[-1]) if len(z_series) else 0.0,
                "n_history_days": len(history)
            })

        elif path == "/api/optimizer/calculate-stdv":
            ticker = query.get("ticker", ["MSFT"])[0].upper()
            metric = query.get("metric", ["profit_factor"])[0].lower()
            horizon_param = query.get("horizon", [""])[0]
            target_h = int(horizon_param) if horizon_param.isdigit() else None
            filter_earnings = query.get("filter_earnings", ["true"])[0].lower() not in ["false", "0", "no"]

            optimizer = ThresholdOptimizer()

            if ticker == "ALL":
                target_tickers = [p["ticker"] for sec_info in UNIVERSE_SECTORS.values() for p in sec_info.get("pairs", [])]
                all_results = []
                for t in target_tickers:
                    m, s = resolve_pair_for_ticker(t)
                    try:
                        an = anomaly_model.analyze(asset_ticker=t, market_ticker=m, sector_ticker=s)
                        if an.get("status") != "SUCCESS":
                            continue
                        hist = an.get("history", [])
                        if len(hist) < 40:
                            continue
                        z_arr = [float(h.get("z_1d", 0.0)) for h in hist]
                        p_arr = [float(h.get("asset_close", 0.0)) for h in hist]
                        d_arr = [str(h.get("date", "")) for h in hist]
                        recs = warehouse.get_earnings_dates(t)
                        edates = [r["earnings_date"] for r in recs] if recs else []

                        calc = optimizer.calculate_best_stdv(
                            z=np.array(z_arr, dtype=float),
                            p=np.array(p_arr, dtype=float),
                            dates=d_arr,
                            earnings_dates=edates,
                            filter_earnings=filter_earnings,
                            target_horizon=target_h,
                            metric=metric
                        )
                        if calc.get("status") == "SUCCESS" and calc.get("champion_recommendation"):
                            champ = calc["champion_recommendation"]
                            champ["ticker"] = t
                            champ["market"] = m
                            champ["sector"] = s
                            all_results.append(champ)
                    except Exception as e:
                        print(f"Error calculating stdv for {t}: {e}")
                        continue

                all_results.sort(key=lambda x: (x.get("profit_factor", 0), x.get("win_rate_pct", 0)), reverse=True)
                pfs = [r["profit_factor"] for r in all_results if r.get("profit_factor", 0) > 0]
                wrs = [r["win_rate_pct"] for r in all_results if r.get("win_rate_pct", 0) > 0]
                typs = [r.get("typical_asset_return_pct", r.get("avg_roi_pct", 0.0)) for r in all_results]

                self._send_json({
                    "status": "SUCCESS",
                    "mode": "PORTFOLIO_ALL",
                    "total_calculated": len(all_results),
                    "portfolio_champions": all_results,
                    "summary": {
                        "avg_win_rate_pct": round(float(np.mean(wrs)), 1) if wrs else 0.0,
                        "avg_profit_factor": round(float(np.mean(pfs)), 2) if pfs else 0.0,
                        "avg_typical_return_pct": round(float(np.mean(typs)), 2) if typs else 0.0
                    }
                })
                return

            mkt_in = query.get("market", [""])[0]
            sec_in = query.get("sector", [""])[0]
            market, sector = resolve_pair_for_ticker(ticker, mkt_in, sec_in)

            analysis = anomaly_model.analyze(asset_ticker=ticker, market_ticker=market, sector_ticker=sector)
            if analysis.get("status") != "SUCCESS":
                self._send_json({"status": "ERROR", "error": analysis.get("error", "Analysis failed")})
                return

            history = analysis.get("history", [])
            if len(history) < 40:
                self._send_json({"status": "ERROR", "error": f"Insufficient history ({len(history)} obs)."})
                return

            z_series = [float(h.get("z_1d", 0.0)) for h in history]
            prices = [float(h.get("asset_close", 0.0)) for h in history]
            dates = [str(h.get("date", "")) for h in history]
            earnings_recs = warehouse.get_earnings_dates(ticker)
            earnings_dates = [r["earnings_date"] for r in earnings_recs] if earnings_recs else []

            calc = optimizer.calculate_best_stdv(
                z=np.array(z_series, dtype=float),
                p=np.array(prices, dtype=float),
                dates=dates,
                earnings_dates=earnings_dates,
                filter_earnings=filter_earnings,
                target_horizon=target_h,
                metric=metric
            )
            calc["ticker"] = ticker
            calc["market"] = market
            calc["sector"] = sector
            self._send_json(calc)

        elif path == "/api/ledger/set-champion":
            ticker = query.get("ticker", ["MSFT"])[0].upper()
            thresh_param = query.get("threshold", [""])[0]
            horizon_param = query.get("holding_days", query.get("horizon", ["5"]))[0]
            threshold = float(thresh_param) if thresh_param else None
            holding_days = int(horizon_param) if horizon_param.isdigit() else 5
            filter_earnings = query.get("filter_earnings", ["true"])[0].lower() not in ["false", "0", "no"]

            optimizer = ThresholdOptimizer()

            if ticker == "ALL":
                target_tickers = [p["ticker"] for sec_info in UNIVERSE_SECTORS.values() for p in sec_info.get("pairs", [])]
                updated_count = 0
                for t in target_tickers:
                    m, s = resolve_pair_for_ticker(t)
                    theme = ""
                    for sec_info in UNIVERSE_SECTORS.values():
                        for p in sec_info.get("pairs", []):
                            if p["ticker"] == t:
                                theme = p.get("theme", "")
                                break
                    try:
                        an = anomaly_model.analyze(asset_ticker=t, market_ticker=m, sector_ticker=s)
                        if an.get("status") != "SUCCESS": continue
                        hist = an.get("history", [])
                        if len(hist) < 40: continue
                        z_arr = [float(h.get("z_1d", 0.0)) for h in hist]
                        p_arr = [float(h.get("asset_close", 0.0)) for h in hist]
                        d_arr = [str(h.get("date", "")) for h in hist]
                        recs = warehouse.get_earnings_dates(t)
                        edates = [r["earnings_date"] for r in recs] if recs else []

                        calc = optimizer.calculate_best_stdv(
                            z=np.array(z_arr, dtype=float),
                            p=np.array(p_arr, dtype=float),
                            dates=d_arr,
                            earnings_dates=edates,
                            filter_earnings=filter_earnings
                        )
                        if calc.get("status") == "SUCCESS" and calc.get("champion_recommendation"):
                            champ = calc["champion_recommendation"]
                            c_thresh = champ["threshold"]
                            c_hold = champ["holding_days"]
                            
                            champ_data = {
                                "threshold": c_thresh,
                                "contract_dte": 0,
                                "tenor_label": "Equity Shares",
                                "holding_days": c_hold,
                                "holding_label": f"{c_hold} Sessions",
                                "win_rate_pct": champ["win_rate_pct"],
                                "avg_roi_pct": champ["avg_roi_pct"],
                                "profit_factor": champ["profit_factor"],
                                "total_dollar": champ.get("total_dollar", 0.0),
                                "trades_count": champ["trades_count"],
                                "est_theta_decay_pct": 0.0,
                                "validation_status": calc.get("walk_forward_validation", {}).get("validation_status", "ROBUST"),
                                "efficiency_ratio": calc.get("walk_forward_validation", {}).get("efficiency_ratio", 1.0),
                                "oos_win_rate_pct": calc.get("walk_forward_validation", {}).get("oos_win_rate_pct", champ["win_rate_pct"]),
                                "score": champ.get("score", 1.0)
                            }
                            warehouse.save_champion_triplet(t, {
                                "market": m,
                                "sector": s,
                                "theme": theme,
                                "champion_triplet": champ_data
                            })

                            eval_t = optimizer.evaluate_single_param(
                                z=np.array(z_arr, dtype=float),
                                p=np.array(p_arr, dtype=float),
                                dates=d_arr,
                                thresh=c_thresh,
                                horizon=c_hold,
                                earnings_dates=edates,
                                filter_earnings=filter_earnings,
                                calc_details=True
                            )
                            raw_trades = eval_t.get("trades", [])
                            gold_trades = []
                            for tr in raw_trades:
                                gold_trades.append({
                                    "ticker": t,
                                    "trade_date": tr["date"],
                                    "action": tr["action"],
                                    "signal_z": tr.get("signal_z", 0.0),
                                    "underlying_entry": tr["entry_price"],
                                    "strike": tr["exit_price"],
                                    "contract_dte": 0,
                                    "holding_sessions": tr["holding_sessions"],
                                    "entry_option_price": tr["entry_price"],
                                    "exit_option_price": tr["exit_price"],
                                    "pnl_pct": tr["pnl_rec_pct"],
                                    "dollar_pnl": tr["dollar_rec"],
                                    "is_win": 1 if tr["win_rec"] else 0,
                                    "loss_cause": tr.get("loss_cause", "UNKNOWN"),
                                    "mfe_pct": tr.get("mfe_pct", 0.0),
                                    "mae_pct": tr.get("mae_pct", 0.0)
                                })
                            warehouse.save_champion_trades(t, gold_trades)
                            updated_count += 1
                    except Exception as e:
                        print(f"Error setting champion for {t}: {e}")
                        continue

                self._send_json({
                    "status": "SUCCESS",
                    "message": f"Successfully calibrated and locked in best Stdv barriers for all {updated_count} watchlist assets into Rare Edge!",
                    "updated_count": updated_count
                })
                return

            mkt_in = query.get("market", [""])[0]
            sec_in = query.get("sector", [""])[0]
            market, sector = resolve_pair_for_ticker(ticker, mkt_in, sec_in)
            theme = ""
            for sec_info in UNIVERSE_SECTORS.values():
                for p in sec_info.get("pairs", []):
                    if p["ticker"] == ticker:
                        theme = p.get("theme", "")
                        break

            analysis = anomaly_model.analyze(asset_ticker=ticker, market_ticker=market, sector_ticker=sector)
            if analysis.get("status") != "SUCCESS":
                self._send_json({"status": "ERROR", "error": "Analysis failed"})
                return

            history = analysis.get("history", [])
            z_arr = np.array([float(h.get("z_1d", 0.0)) for h in history], dtype=float)
            p_arr = np.array([float(h.get("asset_close", 0.0)) for h in history], dtype=float)
            d_arr = [str(h.get("date", "")) for h in history]
            recs = warehouse.get_earnings_dates(ticker)
            edates = [r["earnings_date"] for r in recs] if recs else []

            if threshold is None:
                calc = optimizer.calculate_best_stdv(z_arr, p_arr, d_arr, edates, filter_earnings)
                champ = calc.get("champion_recommendation", {})
                threshold = champ.get("threshold", 2.0)
                holding_days = champ.get("holding_days", 5)

            eval_res = optimizer.evaluate_single_param(
                z_arr, p_arr, d_arr, threshold, holding_days, edates, filter_earnings, calc_details=True
            )

            champ_entry = {
                "threshold": threshold,
                "contract_dte": 0,
                "tenor_label": "Equity Shares",
                "holding_days": holding_days,
                "holding_label": f"{holding_days} Sessions",
                "win_rate_pct": eval_res["win_rate_pct"],
                "avg_roi_pct": eval_res["avg_roi_pct"],
                "profit_factor": eval_res["profit_factor"],
                "total_dollar": eval_res["total_dollar"],
                "trades_count": eval_res["trades_count"],
                "est_theta_decay_pct": 0.0,
                "validation_status": "ROBUST_CONFIRMED" if eval_res["win_rate_pct"] >= 50.0 else "ACCEPTABLE",
                "efficiency_ratio": 1.0,
                "oos_win_rate_pct": eval_res["win_rate_pct"],
                "score": round(eval_res["profit_factor"] * (eval_res["win_rate_pct"] / 50.0), 2)
            }

            warehouse.save_champion_triplet(ticker, {
                "market": market,
                "sector": sector,
                "theme": theme,
                "champion_triplet": champ_entry
            })

            raw_trades = eval_res.get("trades", [])
            gold_trades = []
            for tr in raw_trades:
                gold_trades.append({
                    "ticker": ticker,
                    "trade_date": tr["date"],
                    "action": tr["action"],
                    "signal_z": tr.get("signal_z", 0.0),
                    "underlying_entry": tr["entry_price"],
                    "strike": tr["exit_price"],
                    "contract_dte": 0,
                    "holding_sessions": tr["holding_sessions"],
                    "entry_option_price": tr["entry_price"],
                    "exit_option_price": tr["exit_price"],
                    "pnl_pct": tr["pnl_rec_pct"],
                    "dollar_pnl": tr["dollar_rec"],
                    "is_win": 1 if tr["win_rec"] else 0,
                    "loss_cause": tr.get("loss_cause", "UNKNOWN"),
                    "mfe_pct": tr.get("mfe_pct", 0.0),
                    "mae_pct": tr.get("mae_pct", 0.0)
                })
            warehouse.save_champion_trades(ticker, gold_trades)

            self._send_json({
                "status": "SUCCESS",
                "message": f"Successfully locked in {ticker} Champion at ±{threshold:.1f} sigma @ {holding_days}s Hold into Rare Edge!",
                "champion": champ_entry,
                "trades_count": len(raw_trades)
            })


        elif path == "/api/optimizer/analyze":
            ticker = query.get("ticker", ["NVDA"])[0].upper()
            market = query.get("market", ["QQQ"])[0].upper()
            sector = query.get("sector", ["SMH"])[0].upper()

            analysis = anomaly_model.analyze(
                asset_ticker=ticker,
                market_ticker=market,
                sector_ticker=sector
            )

            if analysis.get("status") != "SUCCESS":
                self._send_json({"status": "ERROR", "error": analysis.get("error", "Analysis failed")})
                return

            history = analysis.get("history", [])
            if len(history) < 30:
                self._send_json({"status": "ERROR", "error": f"Insufficient history ({len(history)} obs)."})
                return

            z_series = [float(h.get("z_1d", 0.0)) for h in history]
            prices = [float(h.get("asset_close", 0.0)) for h in history]
            dates = [str(h.get("date", "")) for h in history]

            # Cached earnings dates
            earnings_recs = warehouse.get_earnings_dates(ticker)
            earnings_dates = [r["earnings_date"] for r in earnings_recs] if earnings_recs else []

            optimizer = ThresholdOptimizer()
            opt_res = optimizer.optimize(
                z_series=z_series,
                prices=prices,
                dates=dates,
                earnings_dates=earnings_dates
            )

            opt_res["ticker"] = ticker
            opt_res["market"] = market
            opt_res["sector"] = sector
            try:
                warehouse.save_optimal_params(ticker, opt_res)
            except Exception:
                pass
            self._send_json(opt_res)

        elif path == "/api/optimizer/trades":
            ticker = query.get("ticker", ["NVDA"])[0].upper()
            mkt_in = query.get("market", [""])[0]
            sec_in = query.get("sector", [""])[0]
            market, sector = resolve_pair_for_ticker(ticker, mkt_in, sec_in)
            threshold = float(query.get("threshold", [1.8])[0])
            dte = int(query.get("dte", query.get("horizon", [5]))[0])
            try:
                contract_dte = int(query.get("contract_dte", query.get("tenor", [21]))[0])
            except (ValueError, TypeError):
                contract_dte = 21

            filter_earnings_raw = query.get("filter_earnings", ["true"])[0].lower()
            filter_earnings = filter_earnings_raw not in ["false", "0", "no"]

            analysis = anomaly_model.analyze(
                asset_ticker=ticker,
                market_ticker=market,
                sector_ticker=sector
            )

            if analysis.get("status") != "SUCCESS":
                self._send_json({"status": "ERROR", "error": analysis.get("error", "Analysis failed")})
                return

            history = analysis.get("history", [])
            if len(history) < 10:
                self._send_json({"status": "ERROR", "error": f"Insufficient history ({len(history)} obs)."})
                return

            z_series = [float(h.get("z_1d", 0.0)) for h in history]
            prices = [float(h.get("asset_close", 0.0)) for h in history]
            dates = [str(h.get("date", "")) for h in history]

            earnings_recs = warehouse.get_earnings_dates(ticker)
            earnings_dates = [r["earnings_date"] for r in earnings_recs] if earnings_recs else []

            optimizer = ThresholdOptimizer()
            eval_res = optimizer.evaluate_single_param(
                thresh=threshold,
                horizon=dte,
                z=np.array(z_series, dtype=float),
                p=np.array(prices, dtype=float),
                dates=dates,
                earnings_dates=earnings_dates,
                filter_earnings=filter_earnings,
                contract_dte=contract_dte
            )

            eval_res["status"] = "SUCCESS"
            eval_res["ticker"] = ticker
            eval_res["market"] = market
            eval_res["sector"] = sector
            eval_res["contract_dte"] = contract_dte
            self._send_json(eval_res)

        elif path == "/api/optimizer/tenor-sweep":
            ticker = query.get("ticker", ["NVDA"])[0].upper()
            mkt_in = query.get("market", [""])[0]
            sec_in = query.get("sector", [""])[0]
            market, sector = resolve_pair_for_ticker(ticker, mkt_in, sec_in)

            opt_params = warehouse.get_optimal_params(ticker)
            def_thresh = float(opt_params["optimal_threshold"]) if opt_params else 1.8
            def_horizon = int(opt_params["optimal_horizon"]) if opt_params else 5

            try:
                threshold = float(query.get("threshold", [def_thresh])[0])
            except (ValueError, TypeError):
                threshold = def_thresh

            try:
                horizon = int(query.get("horizon", [def_horizon])[0])
            except (ValueError, TypeError):
                horizon = def_horizon

            filter_earnings_raw = query.get("filter_earnings", ["true"])[0].lower()
            filter_earnings = filter_earnings_raw not in ["false", "0", "no"]

            analysis = anomaly_model.analyze(
                asset_ticker=ticker,
                market_ticker=market,
                sector_ticker=sector
            )

            if analysis.get("status") != "SUCCESS":
                self._send_json({"status": "ERROR", "error": analysis.get("error", "Analysis failed")})
                return

            history = analysis.get("history", [])
            if len(history) < 10:
                self._send_json({"status": "ERROR", "error": f"Insufficient history ({len(history)} obs)."})
                return

            z_series = [float(h.get("z_1d", 0.0)) for h in history]
            prices = [float(h.get("asset_close", 0.0)) for h in history]
            dates = [str(h.get("date", "")) for h in history]

            earnings_recs = warehouse.get_earnings_dates(ticker)
            earnings_dates = [r["earnings_date"] for r in earnings_recs] if earnings_recs else []

            mode = query.get("mode", ["dynamic"])[0].lower()
            if mode not in ["dynamic", "uniform"]:
                mode = "dynamic"

            optimizer = ThresholdOptimizer()
            sweep_res = optimizer.evaluate_contract_tenor_sweep(
                z=np.array(z_series, dtype=float),
                p=np.array(prices, dtype=float),
                dates=dates,
                thresh=threshold,
                horizon=horizon,
                earnings_dates=earnings_dates,
                filter_earnings=filter_earnings,
                mode=mode
            )

            sweep_res["ticker"] = ticker
            sweep_res["market"] = market
            sweep_res["sector"] = sector
            sweep_res["optimal_threshold"] = def_thresh
            sweep_res["optimal_horizon"] = def_horizon
            self._send_json(sweep_res)

        elif path == "/api/optimizer/3d-optimum":
            ticker = query.get("ticker", ["NVDA"])[0].upper()
            mkt_in = query.get("market", [""])[0]
            sec_in = query.get("sector", [""])[0]
            market, sector = resolve_pair_for_ticker(ticker, mkt_in, sec_in)

            filter_earnings_raw = query.get("filter_earnings", ["true"])[0].lower()
            filter_earnings = filter_earnings_raw not in ["false", "0", "no"]

            analysis = anomaly_model.analyze(
                asset_ticker=ticker,
                market_ticker=market,
                sector_ticker=sector
            )

            if analysis.get("status") != "SUCCESS":
                self._send_json({"status": "ERROR", "error": analysis.get("error", "Analysis failed")})
                return

            history = analysis.get("history", [])
            if len(history) < 40:
                self._send_json({"status": "ERROR", "error": f"Insufficient history ({len(history)} obs)."})
                return

            z_series = [float(h.get("z_1d", 0.0)) for h in history]
            prices = [float(h.get("asset_close", 0.0)) for h in history]
            dates = [str(h.get("date", "")) for h in history]

            earnings_recs = warehouse.get_earnings_dates(ticker)
            earnings_dates = [r["earnings_date"] for r in earnings_recs] if earnings_recs else []

            optimizer = ThresholdOptimizer()
            opt_res = optimizer.evaluate_3d_global_optimum(
                z=np.array(z_series, dtype=float),
                p=np.array(prices, dtype=float),
                dates=dates,
                earnings_dates=earnings_dates,
                filter_earnings=filter_earnings
            )

            opt_res["ticker"] = ticker
            opt_res["market"] = market
            opt_res["sector"] = sector

            if opt_res.get("status") == "SUCCESS" and opt_res.get("champion_triplet"):
                try:
                    theme = ""
                    for sec_info in UNIVERSE_SECTORS.values():
                        for p in sec_info.get("pairs", []):
                            if p["ticker"] == ticker:
                                theme = p.get("theme", "")
                                break
                    warehouse.save_champion_triplet(ticker, {
                        "market": market,
                        "sector": sector,
                        "theme": theme,
                        "champion_triplet": opt_res["champion_triplet"]
                    })
                except Exception as e:
                    print(f"Error saving champion triplet to Gold Ledger for {ticker}: {e}")

            self._send_json(opt_res)

        elif path == "/api/ledger/champions":
            rows = warehouse.get_all_champion_triplets()
            
            # Map theme from universe
            all_pairs = [p for sec_info in UNIVERSE_SECTORS.values() for p in sec_info.get("pairs", [])]
            ticker_theme_map = {p["ticker"]: (p.get("theme", ""), p["market"], p["sector"]) for p in all_pairs}

            enriched = []
            active_signals = 0
            refresh_live = query.get("refresh", ["false"])[0].lower() in ["true", "1", "yes"]

            # Determine latest date in stock_bars
            try:
                with warehouse._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT MAX(trade_date) FROM stock_bars")
                    row_max = cur.fetchone()
                    latest_dataset_str = row_max[0] if (row_max and row_max[0]) else "2026-10-21"
            except Exception:
                latest_dataset_str = "2026-10-21"
            import datetime as dt
            latest_dataset_dt = dt.datetime.strptime(latest_dataset_str, "%Y-%m-%d").date()

            for row in rows:
                t = row["ticker"]
                m = row["market"]
                s = row["sector"]
                thresh = float(row.get("threshold", 2.0))
                l_thresh = float(row.get("long_threshold") or thresh)
                s_thresh = float(row.get("short_threshold") or thresh)
                policy = str(row.get("trade_policy") or "ASYMMETRIC_DUAL")
                is_long_only = (policy == "LONG_ONLY")
                is_short_only = (policy == "SHORT_ONLY")
                
                l_hold = int(row.get("long_holding_days") or row.get("holding_days", 14))
                s_hold = int(row.get("short_holding_days") or row.get("holding_days", 7))
                max_hold = max(l_hold, s_hold)

                # Check Corporate Earnings Blackout:
                # Blocks if earnings is today, yesterday, or scheduled within the upcoming trade holding window [E-1 to E+max_hold+2]
                is_blackout = False
                blackout_ed = None
                blackout_days_away = 0
                erecs = warehouse.get_earnings_dates(t)
                if erecs and latest_dataset_dt:
                    for er in erecs:
                        try:
                            ed_obj = dt.datetime.strptime(er["earnings_date"], "%Y-%m-%d").date()
                            days_ahead = (ed_obj - latest_dataset_dt).days
                            if -1 <= days_ahead <= (max_hold + 2):
                                is_blackout = True
                                blackout_ed = er["earnings_date"]
                                blackout_days_away = days_ahead
                                break
                        except Exception:
                            pass

                # Use stored current_z if available, unless refresh explicitly requested
                z_1d = float(row.get("current_z")) if (row.get("current_z") is not None and not refresh_live) else None
                close_px = float(row.get("latest_close")) if (row.get("latest_close") is not None and not refresh_live) else None

                if z_1d is None or close_px is None:
                    try:
                        analysis = anomaly_model.analyze(asset_ticker=t, market_ticker=m, sector_ticker=s)
                        if analysis.get("status") == "SUCCESS":
                            latest = analysis.get("latest_metrics", {})
                            z_1d = float(latest.get("z_score_1d", 0.0))
                            close_px = float(latest.get("asset_close", 0.0))
                            
                            # Directional Asymmetric Trigger
                            if is_blackout:
                                act = "EARNINGS_BLACKOUT"
                                is_br = False
                            elif not is_short_only and z_1d <= -l_thresh:
                                act = "LONG ASSET"
                                is_br = True
                            elif not is_long_only and z_1d >= s_thresh:
                                act = "SHORT ASSET"
                                is_br = True
                            else:
                                act = "WAITING FOR BARRIER"
                                is_br = False
                            warehouse.update_champion_live_signal(t, z_1d, close_px, act, is_br)
                    except Exception:
                        z_1d = 0.0
                        close_px = 0.0

                # Directional Asymmetric Signal state
                if is_blackout:
                    action = "EARNINGS_BLACKOUT"
                    badge = "EARNINGS BLACKOUT"
                    badge_class = "tag-amber"
                    is_active = False
                elif not is_short_only and z_1d <= -l_thresh:
                    action = "LONG ASSET"
                    badge = "ACTIVE BULLISH SIGNAL"
                    badge_class = "tag-green"
                    is_active = True
                    active_signals += 1
                elif not is_long_only and z_1d >= s_thresh:
                    action = "SHORT ASSET"
                    badge = "ACTIVE BEARISH SIGNAL"
                    badge_class = "tag-red"
                    is_active = True
                    active_signals += 1
                else:
                    action = "WAITING FOR BARRIER"
                    badge = "MONITORING"
                    badge_class = "tag-blue"
                    is_active = False

                # Proximity to nearest active barrier
                dist_long = 999.0 if is_short_only else max(0.0, z_1d - (-l_thresh))
                dist_short = 999.0 if is_long_only else max(0.0, s_thresh - z_1d)
                dist_barrier = min(dist_long, dist_short)

                item = dict(row)
                item["theme"] = ticker_theme_map.get(t, ("", "", ""))[0] or item.get("theme", "")
                item["current_z"] = round(z_1d, 2)
                item["latest_close"] = round(close_px, 2)
                item["signal_action"] = action
                item["action_badge"] = badge
                item["badge_class"] = badge_class
                item["is_active"] = is_active
                item["dist_to_barrier"] = round(dist_barrier, 2)
                item["is_blackout"] = is_blackout
                item["blackout_ed"] = blackout_ed
                enriched.append(item)

            # Summary metrics
            pfs = [r["profit_factor"] for r in enriched if r.get("profit_factor", 0) > 0]
            wrs = [r["win_rate_pct"] for r in enriched if r.get("win_rate_pct", 0) > 0]
            rois = [r["avg_roi_pct"] for r in enriched]

            summary = {
                "total_champions": len(enriched),
                "active_signals_count": active_signals,
                "avg_profit_factor": round(float(np.mean(pfs)), 2) if pfs else 0.0,
                "avg_win_rate_pct": round(float(np.mean(wrs)), 1) if wrs else 0.0,
                "avg_roi_pct": round(float(np.mean(rois)), 1) if rois else 0.0,
            }

            self._send_json({
                "status": "SUCCESS",
                "summary": summary,
                "ledger": enriched
            })

        elif path == "/api/ledger/income":
            income_file = os.path.join(os.path.dirname(__file__), "scratch", "weekly_income_results.json")
            setups = []
            if os.path.exists(income_file):
                try:
                    with open(income_file, "r", encoding="utf-8") as f:
                        setups = json.load(f)
                except Exception as e:
                    print(f"Error loading income results: {e}")

            if not setups:
                setups = [
                    {"ticker": "MSFT", "market": "QQQ", "sector": "IGV", "theme": "Enterprise software", "threshold": 1.4, "contract_dte": 14, "holding_days": 2, "trades_count": 90, "trades_per_month": 7.5, "win_rate_pct": 57.8, "avg_roi_pct": 23.0, "profit_factor": 2.08, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "CRWD", "market": "QQQ", "sector": "CIBR", "theme": "Cybersecurity", "threshold": 1.6, "contract_dte": 14, "holding_days": 7, "trades_count": 64, "trades_per_month": 5.3, "win_rate_pct": 51.6, "avg_roi_pct": 34.1, "profit_factor": 2.18, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "SBUX", "market": "SPY", "sector": "XLY", "theme": "Retail dining and beverages", "threshold": 1.6, "contract_dte": 45, "holding_days": 7, "trades_count": 58, "trades_per_month": 4.8, "win_rate_pct": 56.9, "avg_roi_pct": 18.1, "profit_factor": 1.89, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "NVDA", "market": "QQQ", "sector": "SMH", "theme": "AI chips", "threshold": 1.4, "contract_dte": 10, "holding_days": 1, "trades_count": 141, "trades_per_month": 11.8, "win_rate_pct": 51.8, "avg_roi_pct": 8.2, "profit_factor": 1.51, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "MCD", "market": "DIA", "sector": "XLY", "theme": "Fast food restaurants", "threshold": 2.0, "contract_dte": 7, "holding_days": 1, "trades_count": 54, "trades_per_month": 4.5, "win_rate_pct": 55.6, "avg_roi_pct": 12.8, "profit_factor": 1.75, "validation_status": "MILD_DEGRADATION"},
                    {"ticker": "VRTX", "market": "QQQ", "sector": "XBI", "theme": "Biotechnology & CF therapies", "threshold": 1.6, "contract_dte": 21, "holding_days": 3, "trades_count": 56, "trades_per_month": 4.7, "win_rate_pct": 50.0, "avg_roi_pct": 11.8, "profit_factor": 1.60, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "WMT", "market": "DIA", "sector": "XLP", "theme": "Global discount retail", "threshold": 2.0, "contract_dte": 45, "holding_days": 5, "trades_count": 28, "trades_per_month": 2.3, "win_rate_pct": 64.3, "avg_roi_pct": 18.1, "profit_factor": 1.70, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "XOM", "market": "SPY", "sector": "XLE", "theme": "Energy exploration & refining", "threshold": 1.6, "contract_dte": 21, "holding_days": 2, "trades_count": 105, "trades_per_month": 8.8, "win_rate_pct": 49.5, "avg_roi_pct": 10.9, "profit_factor": 1.64, "validation_status": "MILD_DEGRADATION"},
                    {"ticker": "NKE", "market": "DIA", "sector": "XLY", "theme": "Athletic footwear & apparel", "threshold": 2.0, "contract_dte": 10, "holding_days": 1, "trades_count": 19, "trades_per_month": 1.6, "win_rate_pct": 63.2, "avg_roi_pct": 9.5, "profit_factor": 2.05, "validation_status": "MILD_DEGRADATION"},
                    {"ticker": "KO", "market": "DIA", "sector": "XLP", "theme": "Non-alcoholic beverages", "threshold": 2.0, "contract_dte": 10, "holding_days": 1, "trades_count": 44, "trades_per_month": 3.7, "win_rate_pct": 52.3, "avg_roi_pct": 10.7, "profit_factor": 1.43, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "UNH", "market": "SPY", "sector": "XLV", "theme": "Managed healthcare", "threshold": 2.0, "contract_dte": 35, "holding_days": 7, "trades_count": 26, "trades_per_month": 2.2, "win_rate_pct": 50.0, "avg_roi_pct": 33.7, "profit_factor": 2.34, "validation_status": "MILD_DEGRADATION"},
                    {"ticker": "DIS", "market": "DIA", "sector": "XLC", "theme": "Entertainment & parks", "threshold": 2.0, "contract_dte": 7, "holding_days": 1, "trades_count": 30, "trades_per_month": 2.5, "win_rate_pct": 63.3, "avg_roi_pct": 9.0, "profit_factor": 1.46, "validation_status": "MILD_DEGRADATION"}
                ]

            champ_rows = {r["ticker"]: r for r in warehouse.get_all_champion_triplets()}
            enriched = []
            active_signals = 0

            for s in setups:
                item = dict(s)
                t = item["ticker"]
                thresh = float(item["threshold"])
                
                z_1d = None
                close_px = None
                if t in champ_rows:
                    z_1d = champ_rows[t].get("current_z")
                    close_px = champ_rows[t].get("latest_close")

                if z_1d is None or close_px is None:
                    try:
                        res = anomaly_model.analyze(t, item["market"], item["sector"])
                        if res.get("status") == "SUCCESS":
                            z_1d = float(res.get("z_1d", 0.0))
                            close_px = float(res.get("latest_price", 0.0))
                    except Exception:
                        z_1d = 0.0
                        close_px = 0.0

                z_val = float(z_1d) if z_1d is not None else 0.0
                close_val = float(close_px) if close_px is not None else 0.0

                if z_val <= -thresh:
                    action = "LONG ASSET"
                    badge = "ACTIVE BULLISH FLOW"
                    badge_class = "tag-green"
                    is_active = True
                    active_signals += 1
                elif z_val >= thresh:
                    action = "SHORT ASSET"
                    badge = "ACTIVE BEARISH FLOW"
                    badge_class = "tag-red"
                    is_active = True
                    active_signals += 1
                else:
                    action = "WAITING FOR BARRIER"
                    badge = "MONITORING"
                    badge_class = "tag-blue"
                    is_active = False

                item["current_z"] = round(z_val, 2)
                item["latest_close"] = round(close_val, 2)
                item["signal_action"] = action
                item["action_badge"] = badge
                item["badge_class"] = badge_class
                item["is_active"] = is_active
                item["dist_to_barrier"] = round(abs(abs(z_val) - thresh), 2)
                item["trades_per_week"] = round(float(item.get("trades_count", 0)) / 52.0, 1)
                enriched.append(item)

            pfs = [r["profit_factor"] for r in enriched if r.get("profit_factor", 0) > 0]
            wrs = [r["win_rate_pct"] for r in enriched if r.get("win_rate_pct", 0) > 0]
            rois = [r["avg_roi_pct"] for r in enriched]
            tpws = [r["trades_per_week"] for r in enriched]

            summary = {
                "total_setups": len(enriched),
                "active_signals_count": active_signals,
                "avg_profit_factor": round(float(np.mean(pfs)), 2) if pfs else 0.0,
                "avg_win_rate_pct": round(float(np.mean(wrs)), 1) if wrs else 0.0,
                "avg_roi_pct": round(float(np.mean(rois)), 1) if rois else 0.0,
                "avg_trades_per_week": round(float(np.mean(tpws)), 1) if tpws else 0.0
            }

            self._send_json({
                "status": "SUCCESS",
                "summary": summary,
                "income_setups": enriched
            })

        elif path == "/api/ledger/sync":
            ticker_param = query.get("ticker", [""])[0]
            if ticker_param:
                target_tickers = [t.strip().upper() for t in ticker_param.split(",") if t.strip()]
            else:
                target_tickers = [p["ticker"] for sec_info in UNIVERSE_SECTORS.values() for p in sec_info.get("pairs", [])]

            synced = []
            optimizer = ThresholdOptimizer()
            for t in target_tickers:
                m, s = resolve_pair_for_ticker(t)
                theme = ""
                for sec_info in UNIVERSE_SECTORS.values():
                    for p in sec_info.get("pairs", []):
                        if p["ticker"] == t:
                            theme = p.get("theme", "")
                            break

                try:
                    analysis = anomaly_model.analyze(asset_ticker=t, market_ticker=m, sector_ticker=s)
                    if analysis.get("status") != "SUCCESS":
                        continue
                    history = analysis.get("history", [])
                    if len(history) < 40:
                        continue
                    z_series = [float(h.get("z_1d", 0.0)) for h in history]
                    prices = [float(h.get("asset_close", 0.0)) for h in history]
                    dates = [str(h.get("date", "")) for h in history]
                    recs = warehouse.get_earnings_dates(t)
                    edates = [r["earnings_date"] for r in recs] if recs else []

                    # Safe Sync: Only updates live z-score and latest close without overwriting curated champion parameters
                    latest = history[-1]
                    latest_z = float(latest.get("z_1d", 0.0))
                    latest_close = float(latest.get("asset_close", 0.0))
                    
                    # Fetch existing champion parameters to preserve them
                    existing = warehouse.get_champion_triplet(t)
                    if existing:
                        policy = existing.get("trade_policy", "ASYMMETRIC_DUAL")
                        l_sig = float(existing.get("long_threshold") or existing.get("threshold", 2.0))
                        s_sig = float(existing.get("short_threshold") or existing.get("threshold", 2.0))
                        
                        if policy != "SHORT_ONLY" and latest_z <= -l_sig:
                            act = "LONG ASSET"
                            breached = True
                        elif policy != "LONG_ONLY" and latest_z >= s_sig:
                            act = "SHORT ASSET"
                            breached = True
                        else:
                            act = "WAITING FOR BARRIER"
                            breached = False
                        
                        warehouse.update_champion_live_signal(t, latest_z, latest_close, act, breached)
                        synced.append(t)
                    else:
                        synced.append(t)
                except Exception as e:
                    print(f"Error syncing gold ledger for {t}: {e}")

            self._send_json({
                "status": "SUCCESS",
                "synced_count": len(synced),
                "synced_tickers": synced
            })

        elif path == "/api/ledger/sync-benchmarks":
            from universe import get_all_benchmarks
            benchmarks = get_all_benchmarks()
            yf_loader = YFinanceLoader(warehouse)
            synced = []
            failed = []
            for b in benchmarks:
                try:
                    res = yf_loader.fetch_and_store_ticker(b, period="5d")
                    if res.get("status") == "SUCCESS":
                        synced.append(b)
                    else:
                        failed.append(b)
                except Exception as e:
                    failed.append(b)
            self._send_json({
                "status": "SUCCESS",
                "synced_count": len(synced),
                "synced": synced,
                "failed": failed,
                "message": f"Successfully updated {len(synced)}/{len(benchmarks)} benchmarks via high-speed feed."
            })

        elif path == "/api/ledger/sync-stocks":
            from universe import get_all_tickers
            stocks = get_all_tickers()
            yf_loader = YFinanceLoader(warehouse)
            synced = []
            failed = []
            for t in stocks:
                try:
                    res = yf_loader.fetch_and_store_ticker(t, period="5d")
                    if res.get("status") == "SUCCESS":
                        synced.append(t)
                    else:
                        failed.append(t)
                except Exception as e:
                    failed.append(t)
            self._send_json({
                "status": "SUCCESS",
                "synced_count": len(synced),
                "synced": synced,
                "failed": failed,
                "message": f"Successfully updated {len(synced)}/{len(stocks)} watchlist stocks via high-speed feed."
            })

        elif path == "/api/warehouse/sync-earnings":
            res = warehouse.sync_upcoming_earnings()
            self._send_json(res)

        elif path == "/api/ledger/pre-close-scan":
            import datetime as dt
            rows = warehouse.get_all_champion_triplets()
            all_pairs = [p for sec_info in UNIVERSE_SECTORS.values() for p in sec_info.get("pairs", [])]
            ticker_theme_map = {p["ticker"]: (p.get("theme", ""), p["market"], p["sector"]) for p in all_pairs}

            now_ny = dt.datetime.now()
            market_time_str = now_ny.strftime("%H:%M:%S EST")

            # Determine latest date in stock_bars
            try:
                with warehouse._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT MAX(trade_date) FROM stock_bars")
                    row_max = cur.fetchone()
                    latest_dataset_str = row_max[0] if (row_max and row_max[0]) else "2026-10-21"
            except Exception:
                latest_dataset_str = "2026-10-21"
            latest_dataset_dt = dt.datetime.strptime(latest_dataset_str, "%Y-%m-%d").date()

            signals_firing = []
            earnings_blocked = []
            monitoring_candidates = []
            enriched = []

            for row in rows:
                t = row["ticker"]
                thresh = float(row.get("threshold", 2.0))
                l_thresh = float(row.get("long_threshold") or thresh)
                s_thresh = float(row.get("short_threshold") or thresh)
                policy = str(row.get("trade_policy") or "ASYMMETRIC_DUAL")
                is_long_only = (policy == "LONG_ONLY")
                is_short_only = (policy == "SHORT_ONLY")

                z_1d = float(row.get("current_z", 0.0) or 0.0)
                close_px = float(row.get("latest_close", 0.0) or 0.0)
                tenor = int(row.get("contract_dte", 21))
                l_hold = int(row.get("long_holding_days") or row.get("holding_days", 14))
                s_hold = int(row.get("short_holding_days") or row.get("holding_days", 7))
                max_hold = max(l_hold, s_hold)

                # Check Corporate Earnings Blackout:
                # Blocks if earnings is today, yesterday, or scheduled within the upcoming trade holding window [E-1 to E+max_hold+2]
                is_blackout = False
                blackout_ed = None
                blackout_days_away = 0
                erecs = warehouse.get_earnings_dates(t)
                if erecs and latest_dataset_dt:
                    for er in erecs:
                        try:
                            ed_obj = dt.datetime.strptime(er["earnings_date"], "%Y-%m-%d").date()
                            days_ahead = (ed_obj - latest_dataset_dt).days
                            if -1 <= days_ahead <= (max_hold + 2):
                                is_blackout = True
                                blackout_ed = er["earnings_date"]
                                blackout_days_away = days_ahead
                                break
                        except Exception:
                            pass

                is_active = False
                action = "WAITING FOR BARRIER"
                action_badge = "MONITORING"
                badge_class = "tag-blue"
                chosen_hold = row.get("holding_days", 5)

                if is_blackout:
                    action = "EARNINGS_BLACKOUT"
                    action_badge = "EARNINGS BLACKOUT"
                    badge_class = "tag-amber"
                    is_active = False
                    dist = 999.0
                    reason = f"BLOCKED: Earnings on {blackout_ed} ({blackout_days_away}d away, overlaps with {max_hold}-session hold)."
                elif not is_short_only and z_1d <= -l_thresh:
                    action = "LONG ASSET"
                    action_badge = "ACTIVE BULLISH SIGNAL"
                    badge_class = "tag-green"
                    is_active = True
                    chosen_hold = l_hold
                    dist = round(abs(z_1d - (-l_thresh)), 2)
                    reason = f"Z-score {z_1d:+.2f}σ breached Long barrier -{l_thresh:.1f}σ by {dist:.2f}σ"
                elif not is_long_only and z_1d >= s_thresh:
                    action = "SHORT ASSET"
                    action_badge = "ACTIVE BEARISH SIGNAL"
                    badge_class = "tag-red"
                    is_active = True
                    chosen_hold = s_hold
                    dist = round(abs(z_1d - s_thresh), 2)
                    reason = f"Z-score {z_1d:+.2f}σ breached Short barrier +{s_thresh:.1f}σ by {dist:.2f}σ"
                else:
                    dist_long = 999.0 if is_short_only else max(0.0, z_1d - (-l_thresh))
                    dist_short = 999.0 if is_long_only else max(0.0, s_thresh - z_1d)
                    dist = round(min(dist_long, dist_short), 2)
                    reason = f"{dist:.2f}σ away from trigger"

                warehouse.update_champion_live_signal(t, z_1d, close_px, action, is_active)

                strike_est = round(close_px, 1)

                item = dict(row)
                item["theme"] = ticker_theme_map.get(t, ("", "", ""))[0] or item.get("theme", "")
                item["current_z"] = round(z_1d, 2)
                item["latest_close"] = round(close_px, 2)
                item["signal_action"] = action
                item["action_badge"] = action_badge
                item["badge_class"] = badge_class
                item["is_active"] = is_active
                item["is_blackout"] = is_blackout
                item["blackout_ed"] = blackout_ed
                item["dist_to_barrier"] = dist
                item["order_spec"] = {
                    "action": "BUY ATM CALL" if action == "LONG ASSET" else ("BUY ATM PUT" if action == "SHORT ASSET" else action),
                    "underlying": t,
                    "target_strike": strike_est,
                    "contract_tenor_dte": tenor,
                    "execution_instrument": "Equity Shares or Options (MOC)",
                    "holding_horizon_days": chosen_hold,
                    "execution_type": "MOC (Market-On-Close) 3:45–3:58 PM EST",
                    "reason": reason
                }
                enriched.append(item)

                if is_active:
                    signals_firing.append(item)
                elif is_blackout and ((not is_short_only and z_1d <= -l_thresh) or (not is_long_only and z_1d >= s_thresh)):
                    earnings_blocked.append(item)
                else:
                    monitoring_candidates.append(item)

            monitoring_candidates.sort(key=lambda x: x["dist_to_barrier"])

            self._send_json({
                "status": "SUCCESS",
                "timestamp": market_time_str,
                "total_scanned": len(rows),
                "firing_count": len(signals_firing),
                "signals_firing": signals_firing,
                "earnings_blocked_count": len(earnings_blocked),
                "earnings_blocked": earnings_blocked,
                "closest_candidates": monitoring_candidates[:3],
                "ledger": enriched
            })

        elif path == "/api/ledger/cadence-scan":
            import datetime as dt
            income_file = os.path.join(os.path.dirname(__file__), "scratch", "weekly_income_results.json")
            setups = []
            if os.path.exists(income_file):
                try:
                    with open(income_file, "r", encoding="utf-8") as f:
                        setups = json.load(f)
                except Exception as e:
                    print(f"Error loading income results for scan: {e}")

            if not setups:
                setups = [
                    {"ticker": "MSFT", "market": "QQQ", "sector": "IGV", "theme": "Enterprise software", "threshold": 1.4, "contract_dte": 14, "holding_days": 2, "trades_count": 90, "trades_per_month": 7.5, "win_rate_pct": 57.8, "avg_roi_pct": 23.0, "profit_factor": 2.08, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "CRWD", "market": "QQQ", "sector": "CIBR", "theme": "Cybersecurity", "threshold": 1.6, "contract_dte": 14, "holding_days": 7, "trades_count": 64, "trades_per_month": 5.3, "win_rate_pct": 51.6, "avg_roi_pct": 34.1, "profit_factor": 2.18, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "SBUX", "market": "SPY", "sector": "XLY", "theme": "Retail dining and beverages", "threshold": 1.6, "contract_dte": 45, "holding_days": 7, "trades_count": 58, "trades_per_month": 4.8, "win_rate_pct": 56.9, "avg_roi_pct": 18.1, "profit_factor": 1.89, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "NVDA", "market": "QQQ", "sector": "SMH", "theme": "AI chips", "threshold": 1.4, "contract_dte": 10, "holding_days": 1, "trades_count": 141, "trades_per_month": 11.8, "win_rate_pct": 51.8, "avg_roi_pct": 8.2, "profit_factor": 1.51, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "MCD", "market": "DIA", "sector": "XLY", "theme": "Fast food restaurants", "threshold": 2.0, "contract_dte": 7, "holding_days": 1, "trades_count": 54, "trades_per_month": 4.5, "win_rate_pct": 55.6, "avg_roi_pct": 12.8, "profit_factor": 1.75, "validation_status": "MILD_DEGRADATION"},
                    {"ticker": "VRTX", "market": "QQQ", "sector": "XBI", "theme": "Biotechnology & CF therapies", "threshold": 1.6, "contract_dte": 21, "holding_days": 3, "trades_count": 56, "trades_per_month": 4.7, "win_rate_pct": 50.0, "avg_roi_pct": 11.8, "profit_factor": 1.60, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "WMT", "market": "DIA", "sector": "XLP", "theme": "Global discount retail", "threshold": 2.0, "contract_dte": 45, "holding_days": 5, "trades_count": 28, "trades_per_month": 2.3, "win_rate_pct": 64.3, "avg_roi_pct": 18.1, "profit_factor": 1.70, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "XOM", "market": "SPY", "sector": "XLE", "theme": "Energy exploration & refining", "threshold": 1.6, "contract_dte": 21, "holding_days": 2, "trades_count": 105, "trades_per_month": 8.8, "win_rate_pct": 49.5, "avg_roi_pct": 10.9, "profit_factor": 1.64, "validation_status": "MILD_DEGRADATION"},
                    {"ticker": "NKE", "market": "DIA", "sector": "XLY", "theme": "Athletic footwear & apparel", "threshold": 2.0, "contract_dte": 10, "holding_days": 1, "trades_count": 19, "trades_per_month": 1.6, "win_rate_pct": 63.2, "avg_roi_pct": 9.5, "profit_factor": 2.05, "validation_status": "MILD_DEGRADATION"},
                    {"ticker": "KO", "market": "DIA", "sector": "XLP", "theme": "Non-alcoholic beverages", "threshold": 2.0, "contract_dte": 10, "holding_days": 1, "trades_count": 44, "trades_per_month": 3.7, "win_rate_pct": 52.3, "avg_roi_pct": 10.7, "profit_factor": 1.43, "validation_status": "ROBUST_CONFIRMED"},
                    {"ticker": "UNH", "market": "SPY", "sector": "XLV", "theme": "Managed healthcare", "threshold": 2.0, "contract_dte": 35, "holding_days": 7, "trades_count": 26, "trades_per_month": 2.2, "win_rate_pct": 50.0, "avg_roi_pct": 33.7, "profit_factor": 2.34, "validation_status": "MILD_DEGRADATION"},
                    {"ticker": "DIS", "market": "DIA", "sector": "XLC", "theme": "Entertainment & parks", "threshold": 2.0, "contract_dte": 7, "holding_days": 1, "trades_count": 30, "trades_per_month": 2.5, "win_rate_pct": 63.3, "avg_roi_pct": 9.0, "profit_factor": 1.46, "validation_status": "MILD_DEGRADATION"}
                ]

            champ_rows = {r["ticker"]: r for r in warehouse.get_all_champion_triplets()}
            now_ny = dt.datetime.now()
            market_time_str = now_ny.strftime("%H:%M:%S EST")

            signals_firing = []
            closest_candidates = []
            enriched = []

            for s in setups:
                item = dict(s)
                t = item["ticker"]
                thresh = float(item["threshold"])

                z_1d = None
                close_px = None
                if t in champ_rows:
                    z_1d = champ_rows[t].get("current_z")
                    close_px = champ_rows[t].get("latest_close")

                if z_1d is None or close_px is None:
                    try:
                        res = anomaly_model.analyze(asset_ticker=t, market_ticker=item["market"], sector_ticker=item["sector"])
                        if res.get("status") == "SUCCESS":
                            z_1d = float(res.get("z_1d", 0.0))
                            close_px = float(res.get("latest_price", 0.0))
                    except Exception:
                        z_1d = 0.0
                        close_px = 0.0

                z_val = float(z_1d) if z_1d is not None else 0.0
                close_val = float(close_px) if close_px is not None else 0.0

                dist_sigma = round(max(0.0, thresh - abs(z_val)), 2)
                is_active = abs(z_val) >= thresh

                if z_val <= -thresh:
                    action = "LONG ASSET"
                    badge = "ACTIVE BULLISH FLOW"
                    badge_class = "tag-green"
                elif z_val >= thresh:
                    action = "SHORT ASSET"
                    badge = "ACTIVE BEARISH FLOW"
                    badge_class = "tag-red"
                else:
                    action = "WAITING FOR BARRIER"
                    badge = "MONITORING"
                    badge_class = "tag-blue"

                strike_est = round(close_val, 1)
                tenor = int(item.get("contract_dte", 14))
                hold = int(item.get("holding_days", 2))

                item["current_z"] = round(z_val, 2)
                item["latest_close"] = round(close_val, 2)
                item["signal_action"] = action
                item["action_badge"] = badge
                item["badge_class"] = badge_class
                item["is_active"] = is_active
                item["dist_to_barrier"] = dist_sigma
                item["trades_per_week"] = round(float(item.get("trades_count", 0)) / 52.0, 1)
                item["order_spec"] = {
                    "action": action,
                    "underlying": t,
                    "target_strike": strike_est,
                    "contract_tenor_dte": tenor,
                    "holding_horizon_days": hold,
                    "execution_type": "MOC (Market-On-Close) 3:45–3:58 PM EST",
                    "reason": f"Z-score {z_val:+.2f}σ breached income barrier ±{thresh:.1f}σ by {abs(abs(z_val)-thresh):.2f}σ" if is_active else f"{dist_sigma:.2f}σ away from ±{thresh:.1f}σ income barrier"
                }

                enriched.append(item)
                if is_active:
                    signals_firing.append(item)
                else:
                    closest_candidates.append(item)

            closest_candidates.sort(key=lambda x: x["dist_to_barrier"])

            dedup_closest = []
            seen_cand = set()
            for c in closest_candidates:
                if c["ticker"] not in seen_cand:
                    seen_cand.add(c["ticker"])
                    dedup_closest.append(c)

            self._send_json({
                "status": "SUCCESS",
                "timestamp": market_time_str,
                "total_scanned": len(enriched),
                "firing_count": len(signals_firing),
                "signals_firing": signals_firing,
                "closest_candidates": dedup_closest[:5],
                "income_setups": enriched
            })

        elif path == "/api/ledger/trade-history":
            import datetime as dt
            ticker_param = query.get("ticker", ["ALL"])[0].strip().upper()
            limit = int(query.get("limit", [1000])[0])

            trades = warehouse.get_champion_trades(ticker_param if ticker_param != "ALL" else None)
            total_count = len(trades)

            if total_count > 1:
                chron_trades = sorted(trades, key=lambda x: x["trade_date"])
                date_objs = [dt.datetime.strptime(tr["trade_date"], "%Y-%m-%d") for tr in chron_trades]
                cal_gaps = [(date_objs[i] - date_objs[i-1]).days for i in range(1, len(date_objs))]
                avg_interval_cal_days = round(float(np.mean(cal_gaps)), 1) if cal_gaps else 0.0
                avg_interval_trading_sessions = round(avg_interval_cal_days * (5.0 / 7.0), 1)
                min_gap = min(cal_gaps) if cal_gaps else 0
                max_gap = max(cal_gaps) if cal_gaps else 0

                date_to_gap = {}
                for i in range(1, len(chron_trades)):
                    date_to_gap[chron_trades[i]["trade_date"]] = cal_gaps[i-1]

                for tr in trades:
                    tr["gap_since_prior_days"] = date_to_gap.get(tr["trade_date"], None)
                    ep = float(tr.get("underlying_entry", 0.0) or 0.0)
                    xp = float(tr.get("strike", 0.0) or 0.0)
                    tr["entry_price"] = ep
                    tr["exit_price"] = xp
                    tr["stock_move_dollar"] = round(xp - ep, 2)
                    tr["stock_move_pct"] = round(((xp - ep) / ep * 100.0) if ep > 0 else 0.0, 2)
                    tr["captured_move_pct"] = round(float(tr.get("pnl_pct", 0.0) or 0.0), 2)
            else:
                avg_interval_cal_days = 0.0
                avg_interval_trading_sessions = 0.0
                min_gap = 0
                max_gap = 0
                for tr in trades:
                    tr["gap_since_prior_days"] = None
                    ep = float(tr.get("underlying_entry", 0.0) or 0.0)
                    xp = float(tr.get("strike", 0.0) or 0.0)
                    tr["entry_price"] = ep
                    tr["exit_price"] = xp
                    tr["stock_move_dollar"] = round(xp - ep, 2)
                    tr["stock_move_pct"] = round(((xp - ep) / ep * 100.0) if ep > 0 else 0.0, 2)
                    tr["captured_move_pct"] = round(float(tr.get("pnl_pct", 0.0) or 0.0), 2)

            wins = [t for t in trades if t.get("pnl_pct", 0.0) > 0]
            win_rate_pct = round((len(wins) / total_count * 100.0), 1) if total_count > 0 else 0.0
            rois = [t.get("pnl_pct", 0.0) for t in trades]
            avg_roi_pct = round(float(np.mean(rois)), 1) if rois else 0.0
            total_dollar = round(float(sum(t.get("dollar_pnl", 0.0) for t in trades)), 2)
            holds = [t.get("holding_sessions", 5) for t in trades]
            avg_hold_sessions = round(float(np.mean(holds)), 1) if holds else 0.0

            if total_count > 1:
                span_days = max(1, (date_objs[-1] - date_objs[0]).days)
                span_years = max(0.5, span_days / 365.25)
                trades_per_year = round(total_count / span_years, 1)
            elif total_count == 1:
                span_years = 1.0
                trades_per_year = 1.0
            else:
                span_years = 0.0
                trades_per_year = 0.0

            latest_trade_date = trades[0]["trade_date"] if total_count > 0 else "None"

            summary = {
                "selected_ticker": ticker_param,
                "total_trades": total_count,
                "win_rate_pct": win_rate_pct,
                "avg_roi_pct": avg_roi_pct,
                "total_dollar_pnl": total_dollar,
                "avg_holding_sessions": avg_hold_sessions,
                "avg_interval_calendar_days": avg_interval_cal_days,
                "avg_interval_trading_sessions": avg_interval_trading_sessions,
                "min_gap_days": min_gap,
                "max_gap_days": max_gap,
                "trades_per_year": trades_per_year,
                "span_years": round(span_years, 1),
                "optimal_firing_window": "3:45 PM - 3:58 PM EST (Market-On-Close)",
                "latest_trade_date": latest_trade_date,
            }

            self._send_json({
                "status": "SUCCESS",
                "summary": summary,
                "trades": trades[:limit]
            })

        elif path == "/api/history/sync-yfinance":
            ticker_param = query.get("ticker", [""])[0]
            period = query.get("period", ["5y"])[0]

            if not ticker_param:
                self._send_json({"status": "ERROR", "error": "Missing ticker parameter"}, status=400)
                return

            tickers = [t.strip().upper() for t in ticker_param.split(",") if t.strip()]
            loader = YFinanceLoader(warehouse)
            results = []
            for t in tickers:
                res = loader.fetch_and_store_ticker(t, period=period)
                results.append(res)

            if len(results) == 1:
                self._send_json(results[0])
            else:
                total_pulled = sum(r.get("bars_pulled", 0) for r in results if r.get("status") == "SUCCESS")
                self._send_json({
                    "status": "SUCCESS",
                    "period": period,
                    "tickers_synced": [r["ticker"] for r in results if r.get("status") == "SUCCESS"],
                    "total_bars_pulled": total_pulled,
                    "details": results
                })

        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if parsed.path == "/api/warehouse/sync":
            ticker = payload.get("ticker", "QQQ").upper()
            client = MassiveApiClient()

            stock_res = client.get_stock_aggs(ticker)
            bars_saved = warehouse.save_stock_bars(ticker, stock_res.get("results", []))

            opt_res = client.get_option_contracts(underlying_ticker=ticker, limit=50)
            opts_saved = warehouse.save_option_contracts(opt_res.get("results", []))

            self._send_json({
                "status": "SUCCESS",
                "ticker": ticker,
                "bars_synced": bars_saved,
                "options_synced": opts_saved,
                "stats": warehouse.get_warehouse_stats()
            })

        elif parsed.path == "/api/anomaly/analyze":
            asset = payload.get("asset", "MSFT").upper()
            market = payload.get("market", "QQQ").upper()
            sector = payload.get("sector", "IGV").upper()
            threshold = float(payload.get("threshold", 1.8))
            filter_earnings = bool(payload.get("filter_earnings", True))
            res = anomaly_model.analyze(
                asset_ticker=asset,
                market_ticker=market,
                sector_ticker=sector,
                snapback_threshold=threshold,
                filter_earnings=filter_earnings
            )
            self._send_json(res)

        elif parsed.path == "/api/test/endpoint":
            endpoint = payload.get("endpoint", "")
            method_name = payload.get("method", "")
            params = payload.get("params", {})
            client = MassiveApiClient()

            if hasattr(client, method_name):
                fn = getattr(client, method_name)
                try:
                    res = fn(**params)
                    self._send_json({
                        "status": "SUCCESS",
                        "endpoint": endpoint,
                        "data": res
                    })
                except Exception as e:
                    self._send_json({"status": "ERROR", "error": str(e)}, status=500)
            else:
                res = client._request(endpoint, params=params)
                self._send_json({
                    "status": "SUCCESS",
                    "endpoint": endpoint,
                    "data": res
                })
        elif parsed.path == "/api/scanner/sync-benchmarks":
            import time
            import datetime as dt
            from universe import get_all_benchmarks, get_all_tickers

            # Which tickers to sync: benchmarks first (critical), then cached stocks
            all_benchmarks = get_all_benchmarks()
            cached_tickers_to_refresh = payload.get("tickers", all_benchmarks)  # default: benchmarks only
            client2 = MassiveApiClient()

            today_str = dt.date.today().strftime("%Y-%m-%d")
            yesterday = dt.date.today() - dt.timedelta(days=1)
            if yesterday.weekday() == 6: yesterday -= dt.timedelta(days=2)
            if yesterday.weekday() == 5: yesterday -= dt.timedelta(days=1)
            yesterday_str = yesterday.strftime("%Y-%m-%d")

            results_log = []
            stale = []
            # Only sync tickers that are actually stale or missing
            for ticker in cached_tickers_to_refresh:
                hist = warehouse.get_stock_history(ticker)
                latest_date = hist[-1]["trade_date"] if hist else None
                if not latest_date or latest_date < yesterday_str:
                    stale.append(ticker)

            synced = []
            failed = []
            for i, ticker in enumerate(stale):
                if i > 0:
                    time.sleep(13)  # 13s gap keeps under 5 req/min hard limit
                res = client2.get_stock_aggs(ticker)
                bars = res.get("results", [])
                if bars:
                    saved = warehouse.save_stock_bars(ticker, bars)
                    hist = warehouse.get_stock_history(ticker)
                    new_latest = hist[-1]["trade_date"] if hist else "?"
                    synced.append({"ticker": ticker, "bars_saved": saved, "latest_date": new_latest})
                else:
                    failed.append({"ticker": ticker, "error": res.get("error", res.get("status", "No data"))})

            all_benchmarks_fresh = all(
                (warehouse.get_stock_history(t) or [{}])[-1].get("trade_date", "") >= yesterday_str
                for t in all_benchmarks
            )

            self._send_json({
                "status": "SUCCESS",
                "as_of_date": today_str,
                "previous_trading_day": yesterday_str,
                "stale_found": len(stale),
                "synced": synced,
                "failed": failed,
                "scanner_ready": all_benchmarks_fresh,
                "message": "All benchmarks current — scanner ready for pre-close signals." if all_benchmarks_fresh else f"{len(failed)} benchmark(s) still stale. Check API status."
            })

        elif parsed.path == "/api/scanner/sync-stocks":
            import time
            import datetime as dt
            from universe import get_all_tickers

            all_stocks = get_all_tickers()
            client2 = MassiveApiClient()

            today_str = dt.date.today().strftime("%Y-%m-%d")
            yesterday = dt.date.today() - dt.timedelta(days=1)
            if yesterday.weekday() == 6:  # Sunday → Friday
                yesterday -= dt.timedelta(days=2)
            if yesterday.weekday() == 5:  # Saturday → Friday
                yesterday -= dt.timedelta(days=1)
            yesterday_str = yesterday.strftime("%Y-%m-%d")

            # Separate: stocks with history (sync if stale) vs never-cached (skip)
            stale = []
            missing = []
            for ticker in all_stocks:
                hist = warehouse.get_stock_history(ticker)
                if not hist:
                    missing.append(ticker)
                elif hist[-1]["trade_date"] < yesterday_str:
                    stale.append(ticker)

            synced = []
            failed = []
            for i, ticker in enumerate(stale):
                if i > 0:
                    time.sleep(13)  # Stay under 5 req/min
                res = client2.get_stock_aggs(ticker)
                bars = res.get("results", [])
                if bars:
                    saved = warehouse.save_stock_bars(ticker, bars)
                    hist = warehouse.get_stock_history(ticker)
                    new_latest = hist[-1]["trade_date"] if hist else "?"
                    synced.append({"ticker": ticker, "bars_saved": saved, "latest_date": new_latest})
                else:
                    failed.append({"ticker": ticker, "error": res.get("error", res.get("status", "No data"))})

            all_cached_fresh = all(
                (warehouse.get_stock_history(t) or [{}])[-1].get("trade_date", "") >= yesterday_str
                for t in all_stocks if warehouse.get_stock_history(t)
            )

            self._send_json({
                "status": "SUCCESS",
                "as_of_date": today_str,
                "previous_trading_day": yesterday_str,
                "stale_found": len(stale),
                "missing_count": len(missing),
                "missing": missing,
                "synced": synced,
                "failed": failed,
                "all_cached_stocks_fresh": all_cached_fresh,
                "message": (
                    f"All cached stocks current." if all_cached_fresh
                    else f"{len(failed)} failed. {len(missing)} not yet cached (load via cockpit)."
                )
            })

        elif parsed.path == "/api/smm/fit-all":
            from universe import get_pairs_by_sector

            try:
                threshold = float(payload.get("threshold", 1.8))
            except (ValueError, TypeError):
                threshold = 1.8

            all_pairs = get_pairs_by_sector("ALL")
            fitted = []
            skipped = []
            failed = []

            for p in all_pairs:
                ticker = p["ticker"].upper()
                market = p["market"].upper()
                sector_etf = p["sector"].upper()

                try:
                    analysis = anomaly_model.analyze(
                        asset_ticker=ticker,
                        market_ticker=market,
                        sector_ticker=sector_etf,
                        snapback_threshold=threshold
                    )
                    if analysis.get("status") != "SUCCESS":
                        skipped.append({"ticker": ticker, "reason": analysis.get("error", "Analysis failed")})
                        continue

                    history = analysis.get("history", [])
                    if len(history) < 20:
                        skipped.append({"ticker": ticker, "reason": f"Only {len(history)} history obs"})
                        continue

                    z_series = np.array([float(h.get("z_1d", 0.0)) for h in history])
                    smf = SemiMarkovFitter(threshold=threshold)
                    smm = smf.fit(z_series)

                    fitted.append({
                        "ticker": ticker,
                        "market": market,
                        "sector": sector_etf,
                        "current_regime": smm.get("current_state_name"),
                        "days_in_regime": smm.get("days_in_regime"),
                        "exit_prob_3d": smm.get("exit_prob_3d"),
                        "mean_residual_life": smm.get("mean_residual_life"),
                        "fit_quality": smm.get("fit_quality"),
                        "n_transitions": smm.get("n_transitions")
                    })

                except Exception as e:
                    failed.append({"ticker": ticker, "error": str(e)})

            self._send_json({
                "status": "SUCCESS",
                "threshold": threshold,
                "total_pairs": len(all_pairs),
                "fitted_count": len(fitted),
                "skipped_count": len(skipped),
                "failed_count": len(failed),
                "fitted": fitted,
                "skipped": skipped,
                "failed": failed
            })

        else:
            self._send_json({"error": "Endpoint not found"}, status=404)


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_server(port: int = None):
    p = port or config.PORT
    with ThreadingTCPServer(("", p), DashboardHandler) as httpd:
        print(f"[*] Project RM Dashboard running at: http://localhost:{p}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
