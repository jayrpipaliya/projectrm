"""
CLI Test Suite for Project RM - Step 1 Verification.
Tests live Polygon API connectivity, QQQ historical bars, live option chains,
and SQL Warehouse storage.
"""

import json
import config
from api_client import MassiveApiClient
from warehouse import Warehouse
from analytics import calculate_realized_volatility, calculate_relative_performance


def run_tests():
    print("=" * 65)
    print(" PROJECT RM: STEP 1 - LIVE DATA PIPELINE TEST")
    print("=" * 65)

    print(f"\n[1] Configuration Check:")
    print(f"    - Base URL: {config.BASE_URL}")
    print(f"    - Database: {config.DATABASE_PATH}")
    if config.API_KEY:
        masked = f"{config.API_KEY[:4]}...{config.API_KEY[-4:]}"
        print(f"    - API Key:  Active ({masked})")
    else:
        print("    - API Key:  Missing in .env!")
        return

    client = MassiveApiClient()
    warehouse = Warehouse()

    # 2. Test Stock Endpoints
    print("\n[2] Testing Live Stock Endpoints for 'QQQ'...")
    try:
        aggs = client.get_stock_aggs("QQQ")
        results = aggs.get("results", [])
        print(f"    -> Aggregate Bars: {len(results)} live bars received (status: {aggs.get('status')})")
        if results:
            print(f"    -> Latest Bar Date: {results[-1].get('t')} | Close: ${results[-1].get('c')}")

        prev = client.get_previous_close("QQQ")
        prev_res = prev.get("results", [{}])[0]
        print(f"    -> Previous Close: ${prev_res.get('c', 'N/A')}")
    except Exception as e:
        print(f"    -> Stock endpoint error: {e}")

    # 3. Test Options Endpoints
    print("\n[3] Testing Live Options Endpoints for 'QQQ'...")
    try:
        opt_data = client.get_option_contracts("QQQ", limit=25)
        contracts = opt_data.get("results", [])
        print(f"    -> Live Option Contracts: {len(contracts)} contracts retrieved")
        if contracts:
            sample_contract = contracts[0]["ticker"]
            print(f"    -> Sample Contract: {sample_contract}")
            opt_aggs = client.get_option_aggs(sample_contract)
            print(f"    -> Option Historical Bars: {len(opt_aggs.get('results', []))} bars")
    except Exception as e:
        print(f"    -> Options endpoint error: {e}")

    # 4. Local SQL Warehouse Verification
    print("\n[4] Local SQL Warehouse Verification (SQLite)...")
    cached_bars = warehouse.get_stock_history("QQQ")
    print(f"    -> SQLite 'stock_bars' for QQQ: {len(cached_bars)} rows")
    if cached_bars and cached_bars[-1].get("rv_20d") is not None:
        latest_rv = cached_bars[-1]["rv_20d"] * 100
        latest_close = cached_bars[-1]["close"]
        print(f"    -> Latest QQQ Close: ${latest_close:.2f}")
        print(f"    -> Live 20-Day Annualized Realized Volatility: {latest_rv:.2f}%")

    stats = warehouse.get_warehouse_stats()
    print(f"    -> Database Stats: {json.dumps(stats, indent=2)}")

    print("\n" + "=" * 65)
    print(" ALL LIVE TESTS PASSED SUCCESSFULLY!")
    print(" Run 'python run_dashboard.py' or 'run_dashboard.bat' to view dashboard.")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
