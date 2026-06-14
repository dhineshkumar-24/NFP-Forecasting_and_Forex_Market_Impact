import os
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directories
BASE_DIR = Path(__file__).resolve().parent

FRED_API_KEY = os.getenv("FRED_API_KEY", "2a95be3f3ef77f9a3b23c22ae9b17c4f")

START_DATE = "2000-01-01"
END_DATE = None

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_PATH = "data/processed_macro_data.parquet"

FRED_SERIES = {
    "PAYEMS":            ("PAYEMS",            "Total Nonfarm Payrolls (Target)"),
    "ICSA":              ("ICSA",              "Initial Jobless Claims"),
    "CCSA":              ("CCSA",              "Continuing Claims"),
    "UNRATE":            ("UNRATE",            "Unemployment Rate"),
    "CIVPART":           ("CIVPART",           "Labor Force Participation Rate"),
    "EMRATIO":           ("EMRATIO",           "Employment-Population Ratio"),
    "AWHAETP":           ("AWHAETP",           "Average Weekly Hours"),
    "CES0500000003":     ("CES0500000003",     "Average Hourly Earnings"),
    "JTSJOL":            ("JTSJOL",            "JOLTS Job Openings"),
    "JTSHIL":            ("JTSHIL",            "JOLTS Hires"),
    "JTSQUL":            ("JTSQUL",            "JOLTS Quits"),
    "RSAFS":             ("RSAFS",             "Retail Sales"),
    "INDPRO":            ("INDPRO",            "Industrial Production Index"),
    "TCU":               ("TCU",               "Capacity Utilization"),
    "UMCSENT":           ("UMCSENT",           "Consumer Sentiment"),
    "CSCICP03USM665S":   ("CSCICP03USM665S",   "Consumer Confidence Index"),
    "NFCI":              ("NFCI",              "Financial Conditions Index"),
    "ADPMNUSNERSA":      ("ADPMNUSNERSA",      "ADP Private Payrolls (Monthly)"),
    "BSCICP03USM665S":   ("BSCICP03USM665S",   "OECD Business Confidence Index (USA)"),
    "CPIAUCSL":          ("CPIAUCSL",          "CPI All Urban Consumers"),
    "CPILFESL":          ("CPILFESL",          "Core CPI"),
    "PPIACO":            ("PPIACO",            "PPI All Commodities"),
    "WPSFD4131":         ("WPSFD4131",         "Core PPI Finished Goods"),
    "PCEPI":             ("PCEPI",             "PCE Price Index"),
    "PCEPILFE":          ("PCEPILFE",          "Core PCE Price Index"),
    "DGS2":              ("DGS2",              "2-Year Treasury Yield"),
    "DGS10":             ("DGS10",             "10-Year Treasury Yield"),
    "FEDFUNDS":          ("FEDFUNDS",          "Federal Funds Effective Rate"),
    "M2SL":              ("M2SL",              "M2 Money Supply"),
    "TOTALSA":           ("TOTALSA",           "Total Vehicle Sales"),
    "T10Y2Y":            ("T10Y2Y",            "10Y-2Y Yield Spread"),
    "DTWEXBGS":          ("DTWEXBGS",          "Dollar Index (Broad)"),
    "VIXCLS":            ("VIXCLS",            "VIX Volatility Index"),
    "HOUST":             ("HOUST",             "Housing Starts"),
    "PERMIT":            ("PERMIT",            "Building Permits"),
    "DSPIC96":           ("DSPIC96",           "Real Disposable Personal Income"),
    "DGORDER":           ("DGORDER",           "Durable Goods Orders"),
    "AMTMNO":            ("AMTMNO",            "Manufacturers New Orders"),
    "USREC":             ("USREC",             "NBER Recession Indicator"),
}

WEEKLY_SERIES = ["ICSA", "CCSA"]
DAILY_SERIES  = ["DGS2", "DGS10", "T10Y2Y", "DTWEXBGS", "VIXCLS"]

LAG_PERIODS        = [1, 2, 3]
ROLLING_WINDOWS    = [3, 6, 12]
ROLLING_STD_WINDOW = 6
MOMENTUM_WINDOW    = 3
ZSCORE_WINDOW      = 24

def setup_logging(log_level=logging.INFO):
    """
    Sets up a standardized logging configuration.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(BASE_DIR / "nfp_forecaster.log", encoding="utf-8")
        ]
    )
    logger = logging.getLogger("NFPForecaster")
    return logger

# Initialize global logger
logger = setup_logging()

