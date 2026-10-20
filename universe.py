"""
PROJECT RM: UNIVERSE CONFIGURATION
Defines the focused 12-ticker institutional universe across 4 core sectors,
each calibrated against broad market (SPY / QQQ) and specialized industry benchmark ETFs.
"""

UNIVERSE_SECTORS = {
    "tech_cloud": {
        "name": "Tech & Cloud",
        "description": "Enterprise software, AI chips, memory semiconductors, cybersecurity, digital ads & e-commerce",
        "pairs": [
            {"ticker": "MSFT", "market": "QQQ", "sector": "IGV", "theme": "Enterprise software"},
            {"ticker": "NVDA", "market": "QQQ", "sector": "SMH", "theme": "AI chips"},
            {"ticker": "AMD", "market": "QQQ", "sector": "SMH", "theme": "Semiconductors"},
            {"ticker": "MU", "market": "QQQ", "sector": "DRAM", "theme": "Memory & storage chips"},
            {"ticker": "INTC", "market": "QQQ", "sector": "SMH", "theme": "Semiconductor manufacturing & CPUs"},
            {"ticker": "CRWD", "market": "QQQ", "sector": "CIBR", "theme": "Cybersecurity"},
            {"ticker": "META", "market": "QQQ", "sector": "XLC", "theme": "Digital advertising"},
            {"ticker": "AMZN", "market": "QQQ", "sector": "XLY", "theme": "E-commerce and cloud"},
            {"ticker": "ADBE", "market": "QQQ", "sector": "IGV", "theme": "Creative cloud software"}
        ]
    },
    "consumer_discretionary": {
        "name": "Consumer & Autos",
        "description": "Electric vehicles, clean energy, athletic apparel & dining",
        "pairs": [
            {"ticker": "TSLA", "market": "QQQ", "sector": "XLY", "theme": "EV and clean energy"},
            {"ticker": "SBUX", "market": "SPY", "sector": "XLY", "theme": "Retail dining and beverages"},
            {"ticker": "NKE", "market": "DIA", "sector": "XLY", "theme": "Athletic footwear & apparel"},
            {"ticker": "MCD", "market": "DIA", "sector": "XLY", "theme": "Fast food restaurants"}
        ]
    },
    "consumer_staples": {
        "name": "Consumer Staples",
        "description": "Everyday retail and beverage staples",
        "pairs": [
            {"ticker": "KO", "market": "DIA", "sector": "XLP", "theme": "Non-alcoholic beverages"},
            {"ticker": "WMT", "market": "DIA", "sector": "XLP", "theme": "Global discount retail"}
        ]
    },
    "financials": {
        "name": "Financials",
        "description": "Commercial and investment banking",
        "pairs": [
            {"ticker": "JPM", "market": "SPY", "sector": "XLF", "theme": "Commercial banking"}
        ]
    },
    "healthcare": {
        "name": "Healthcare",
        "description": "Pharmaceuticals, managed care & biotechnology",
        "pairs": [
            {"ticker": "LLY", "market": "SPY", "sector": "XLV", "theme": "Pharmaceuticals"},
            {"ticker": "UNH", "market": "SPY", "sector": "XLV", "theme": "Managed healthcare"},
            {"ticker": "VRTX", "market": "QQQ", "sector": "XBI", "theme": "Biotechnology & CF therapies"}
        ]
    },
    "communications": {
        "name": "Media & Entertainment",
        "description": "Global streaming, entertainment & theme parks",
        "pairs": [
            {"ticker": "DIS", "market": "DIA", "sector": "XLC", "theme": "Entertainment & parks"}
        ]
    },
    "energy": {
        "name": "Energy",
        "description": "Energy exploration and refining",
        "pairs": [
            {"ticker": "XOM", "market": "SPY", "sector": "XLE", "theme": "Energy exploration & refining"}
        ]
    }
}

def get_all_tickers():
    """Returns a flat sorted list of all unique target stock tickers in the universe."""
    tickers = set()
    for sector_data in UNIVERSE_SECTORS.values():
        for pair in sector_data["pairs"]:
            tickers.add(pair["ticker"])
    return sorted(list(tickers))

def get_all_benchmarks():
    """Returns all unique market and sector benchmark tickers."""
    benchmarks = set()
    for sector_data in UNIVERSE_SECTORS.values():
        for pair in sector_data["pairs"]:
            benchmarks.add(pair["market"])
            benchmarks.add(pair["sector"])
    return sorted(list(benchmarks))

def get_pairs_by_sector(sector_id="ALL"):
    """Returns the list of pairs for a given sector or all sectors."""
    if sector_id == "ALL":
        all_pairs = []
        for s_id, s_data in UNIVERSE_SECTORS.items():
            for p in s_data["pairs"]:
                pair_copy = dict(p)
                pair_copy["sector_id"] = s_id
                pair_copy["sector_group"] = s_data["name"]
                all_pairs.append(pair_copy)
        return all_pairs
    
    if sector_id in UNIVERSE_SECTORS:
        s_data = UNIVERSE_SECTORS[sector_id]
        pairs = []
        for p in s_data["pairs"]:
            pair_copy = dict(p)
            pair_copy["sector_id"] = sector_id
            pair_copy["sector_group"] = s_data["name"]
            pairs.append(pair_copy)
        return pairs
    return []

def get_pair_for_ticker(ticker):
    """Finds the benchmark configuration for a given ticker."""
    ticker = ticker.upper()
    for s_id, s_data in UNIVERSE_SECTORS.items():
        for p in s_data["pairs"]:
            if p["ticker"] == ticker:
                res = dict(p)
                res["sector_id"] = s_id
                res["sector_group"] = s_data["name"]
                return res
    return None
