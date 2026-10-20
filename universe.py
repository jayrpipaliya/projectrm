"""
PROJECT RM: INSTITUTIONAL 50-TICKER WATCHLIST UNIVERSE CONFIGURATION
Defines the institutional 50-ticker liquid universe across 11 core sectors,
each calibrated against broad market (SPY / QQQ / DIA / IWM) and specialized industry benchmark ETFs.
"""

UNIVERSE_SECTORS = {
    "software_saas": {
        "name": "Enterprise Software & Cloud",
        "description": "Enterprise cloud software, workflow automation, and infrastructure",
        "pairs": [
            {"ticker": "MSFT", "market": "QQQ", "sector": "IGV", "theme": "Enterprise cloud & software"},
            {"ticker": "CRM",  "market": "QQQ", "sector": "IGV", "theme": "Cloud CRM software"},
            {"ticker": "NOW",  "market": "QQQ", "sector": "IGV", "theme": "Workflow cloud automation"},
            {"ticker": "ADBE", "market": "QQQ", "sector": "IGV", "theme": "Creative cloud software"},
            {"ticker": "ORCL", "market": "SPY", "sector": "IGV", "theme": "Database & cloud infrastructure"},
        ]
    },
    "cybersecurity": {
        "name": "Cybersecurity",
        "description": "Network security platforms, zero trust & endpoint protection",
        "pairs": [
            {"ticker": "PANW", "market": "QQQ", "sector": "CIBR", "theme": "Cybersecurity platform architecture"},
            {"ticker": "CRWD", "market": "QQQ", "sector": "CIBR", "theme": "Cybersecurity endpoint & threat intel"},
        ]
    },
    "semiconductors": {
        "name": "Semiconductors & Equipment",
        "description": "AI accelerators, microprocessors, memory chips & fabrication tools",
        "pairs": [
            {"ticker": "NVDA", "market": "QQQ", "sector": "SMH", "theme": "AI chips & GPU compute"},
            {"ticker": "AMD",  "market": "QQQ", "sector": "SMH", "theme": "Advanced micro devices & x86/GPU"},
            {"ticker": "AVGO", "market": "QQQ", "sector": "SMH", "theme": "Custom silicon & networking chips"},
            {"ticker": "QCOM", "market": "QQQ", "sector": "SMH", "theme": "Wireless & mobile semiconductors"},
            {"ticker": "AMAT", "market": "QQQ", "sector": "SMH", "theme": "Semiconductor wafer equipment"},
            {"ticker": "MU",   "market": "QQQ", "sector": "SMH", "theme": "Memory & storage semiconductors"},
        ]
    },
    "tech_hardware": {
        "name": "Consumer Tech & Hardware",
        "description": "Consumer ecosystems, IT hardware & enterprise server architecture",
        "pairs": [
            {"ticker": "AAPL", "market": "QQQ", "sector": "XLK", "theme": "Consumer electronics & ecosystem"},
            {"ticker": "DELL", "market": "SPY", "sector": "XLK", "theme": "Enterprise IT hardware & servers"},
        ]
    },
    "communications": {
        "name": "Digital Advertising, Media & Streaming",
        "description": "Search advertising, social networks, streaming media & theme parks",
        "pairs": [
            {"ticker": "GOOGL", "market": "QQQ", "sector": "XLC", "theme": "Search, digital advertising & cloud"},
            {"ticker": "META",  "market": "QQQ", "sector": "XLC", "theme": "Social platforms & AI advertising"},
            {"ticker": "NFLX",  "market": "QQQ", "sector": "XLC", "theme": "Subscription streaming entertainment"},
            {"ticker": "DIS",   "market": "DIA", "sector": "XLC", "theme": "Entertainment, studios & theme parks"},
            {"ticker": "CMCSA", "market": "SPY", "sector": "XLC", "theme": "Cable broadband & media entertainment"},
        ]
    },
    "consumer_discretionary": {
        "name": "Consumer Discretionary & Dining",
        "description": "E-commerce, EV, retail home improvement, dining & athletic apparel",
        "pairs": [
            {"ticker": "AMZN", "market": "QQQ", "sector": "XLY", "theme": "E-commerce & AWS cloud"},
            {"ticker": "TSLA", "market": "QQQ", "sector": "XLY", "theme": "EV, autonomy & clean energy"},
            {"ticker": "HD",   "market": "DIA", "sector": "XLY", "theme": "Home improvement retail"},
            {"ticker": "NKE",  "market": "DIA", "sector": "XLY", "theme": "Athletic footwear & apparel"},
            {"ticker": "MCD",  "market": "DIA", "sector": "XLY", "theme": "Fast food franchising & dining"},
            {"ticker": "SBUX", "market": "SPY", "sector": "XLY", "theme": "Retail coffee & dining"},
            {"ticker": "TGT",  "market": "SPY", "sector": "XLY", "theme": "Discount department store retail"},
        ]
    },
    "consumer_staples": {
        "name": "Consumer Staples & Wholesale",
        "description": "Household personal care, non-alcoholic beverages & discount groceries",
        "pairs": [
            {"ticker": "PG",   "market": "DIA", "sector": "XLP", "theme": "Household & personal care products"},
            {"ticker": "KO",   "market": "DIA", "sector": "XLP", "theme": "Non-alcoholic beverages"},
            {"ticker": "PEP",  "market": "QQQ", "sector": "XLP", "theme": "Beverages & global snacks"},
            {"ticker": "COST", "market": "QQQ", "sector": "XLP", "theme": "Wholesale club retail"},
            {"ticker": "WMT",  "market": "DIA", "sector": "XLP", "theme": "Discount retail & grocery"},
        ]
    },
    "financials": {
        "name": "Financials & Banking",
        "description": "Money center banking, investment banking, payments & regional lending",
        "pairs": [
            {"ticker": "JPM", "market": "SPY", "sector": "XLF", "theme": "Money center banking & markets"},
            {"ticker": "BAC", "market": "SPY", "sector": "XLF", "theme": "Consumer & commercial banking"},
            {"ticker": "WFC", "market": "SPY", "sector": "XLF", "theme": "Commercial & retail banking"},
            {"ticker": "GS",  "market": "DIA", "sector": "XLF", "theme": "Investment banking & global markets"},
            {"ticker": "MS",  "market": "SPY", "sector": "XLF", "theme": "Wealth management & investment banking"},
            {"ticker": "V",   "market": "DIA", "sector": "XLF", "theme": "Global digital payments network"},
            {"ticker": "USB", "market": "IWM", "sector": "KRE", "theme": "Regional & commercial banking"},
        ]
    },
    "energy": {
        "name": "Energy Exploration & Oilfield Services",
        "description": "Integrated energy, exploration, production & oilfield technology",
        "pairs": [
            {"ticker": "XOM", "market": "SPY", "sector": "XLE", "theme": "Integrated energy & refining"},
            {"ticker": "CVX", "market": "DIA", "sector": "XLE", "theme": "Integrated oil & gas operations"},
            {"ticker": "COP", "market": "SPY", "sector": "XLE", "theme": "Oil & gas exploration & production"},
            {"ticker": "SLB", "market": "SPY", "sector": "OIH", "theme": "Oilfield equipment & technology"},
        ]
    },
    "healthcare": {
        "name": "Healthcare & Biotechnology",
        "description": "Pharmaceuticals, managed care, GLP-1 therapies & biotechnology",
        "pairs": [
            {"ticker": "LLY",  "market": "SPY", "sector": "XLV", "theme": "Pharmaceuticals & GLP-1 therapies"},
            {"ticker": "UNH",  "market": "DIA", "sector": "XLV", "theme": "Managed healthcare & Optum services"},
            {"ticker": "PFE",  "market": "SPY", "sector": "XLV", "theme": "Biopharmaceuticals & vaccines"},
            {"ticker": "VRTX", "market": "QQQ", "sector": "XBI", "theme": "Biotechnology & CF therapies"},
        ]
    },
    "industrials_materials": {
        "name": "Industrials, Aerospace & Mining",
        "description": "Heavy industrial machinery, aerospace defense & global copper mining",
        "pairs": [
            {"ticker": "CAT", "market": "DIA", "sector": "XLI",  "theme": "Heavy machinery & construction equipment"},
            {"ticker": "BA",  "market": "DIA", "sector": "ITA",  "theme": "Commercial aerospace & defense systems"},
            {"ticker": "FCX", "market": "SPY", "sector": "COPX", "theme": "Copper mining & basic materials"},
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
