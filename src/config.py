"""
Central configuration: hyperparameters, FIPS codes, paths, and constants.
All tuneable values live here to keep src/* free of magic numbers.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models_saved"
LOGS_DIR = ROOT_DIR / "logs"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data Sources — California State Datasets (active pipeline)
# ---------------------------------------------------------------------------

# Raw CSV filenames (data/raw/).  WW: CA Wastewater Surveillance export.
# Cases: CA Statewide COVID-19 Cases/Deaths/Tests (daily → resampled to W-WED).
CA_WW_FILENAME    = "California_Wastewater_Surveillance_Data.csv"
CA_CASES_FILENAME = "Statewide_COVID-19_Cases_Deaths_Tests.csv"

# WW signal column to use from the CA dataset (solid track only).
# "Raw Concentration" or "Norm Pmmov" — update based on EDA notebook 03 Section 3 verdict.
CA_WW_SIGNAL_COL  = "Raw Concentration"

# ---------------------------------------------------------------------------
# CDC NWSS Data Sources (archived — superseded by CA state datasets above)
# ---------------------------------------------------------------------------

CDC_NWSS_URL = (
    "https://data.cdc.gov/resource/2ew6-ywp6.json"  # NWSS public dataset endpoint
)
CDC_CASES_URL = "https://data.cdc.gov/resource/pwn4-m3yp.json"  # county-level cases

NWSS_DATE_COL = "sample_collect_date"
CASES_DATE_COL = "submission_date"
TARGET_COL     = "log1p_new_cases"          # log1p-transformed weekly new COVID-19 cases (prediction target)
WW_FEATURE_COL = "log1p_concentration"      # primary wastewater hist_exog feature (log1p of copies/g)
POPULATION_COL = "population_served"
SEWERSHED_COL = "wwtp_id"
COUNTY_COL = "county_fips"

# ---------------------------------------------------------------------------
# Bay Area 9-County FIPS Codes  (State FIPS 06 = California)
# ---------------------------------------------------------------------------

BAY_AREA_FIPS: dict[str, str] = {
    "Alameda":       "06001",
    "Contra Costa":  "06013",
    "Marin":         "06041",
    "Napa":          "06055",
    "San Francisco": "06075",
    "San Mateo":     "06081",
    "Santa Clara":   "06085",
    "Solano":        "06095",
    "Sonoma":        "06097",
}

# Reverse map for lookup
FIPS_TO_COUNTY: dict[str, str] = {v: k for k, v in BAY_AREA_FIPS.items()}

# Single-county validation target (largest, best-instrumented county)
SANTA_CLARA_FIPS: str = "06085"

# 3-county validation set: spatial-temporal synchrony benchmark
# Chosen for high reporting density, strong correlation, and diverse population.
THREE_COUNTY_FIPS: list[str] = ["06075", "06081", "06085"]  # SF, San Mateo, Santa Clara

# Flat list of county names — used by CA dataset loaders (no FIPS needed there)
BAY_AREA_COUNTIES: list[str] = list(BAY_AREA_FIPS.keys())

# High-priority counties for outbreak surveillance (large population / dense transit)
PRIORITY_COUNTIES: list[str] = ["6075", "6085", "6001", "6081"]  # SF, Santa Clara, Alameda, San Mateo


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

# Overlap window: intersection of CA wastewater solid-track and CA cases data.
#   WW solid track earliest: Santa Clara 2020-07-16; SF 2020-11-09; San Mateo 2020-12-08.
#   Remaining 6 counties join the solid track from 2022 (start_padding_enabled handles NaN warmup).
#   Cases dataset ends: 2023-12-19 (hard limit — CA Statewide dataset cutoff).
#   Full window ≈ 180 W-WED periods across 3.5 years — captures 4 distinct outbreak waves:
#     Summer 2020, Winter 2020-21, Delta 2021, Omicron 2022.
DATA_START_DATE = "2020-07-01"  # overlap window start (earliest CA WW solid data)
DATA_END_DATE   = "2023-12-19"  # overlap window end   (last date in CA Cases dataset)

# Split strategy (public-health-informed H = 8 weeks, W-WED anchored):
#   CV window  : ~35 W-WED weeks  (2022-10-05 – 2023-06-07, ~8 folds at step_weeks=4)
#   Holdout    : ~28 W-WED weeks  (2023-06-08 – 2023-12-19; 3.5× H, good evaluation coverage)
#
#   TRAIN_END_DATE = first cutoff for expanding-window CV:
#     Chosen after all 9 counties have solid WW data (last county Napa: 2022-09-26).
#     From DATA_START_DATE to 2022-10-05 ≈ 117 W-WED weeks >> INPUT_SIZE + H = 34 minimum.
#
#   VAL_END_DATE = end of CV window / last expanding-window cutoff:
#     2022-10-05 + 35 × 7 days = 2023-06-07  (Wednesday ✓)
TRAIN_END_DATE = "2022-10-05"   # first CV cutoff  (all 9 counties active; Wednesday)
VAL_END_DATE   = "2023-06-07"   # end of CV window (~35 weeks beyond first cutoff; Wednesday)

OUTLIER_Z_THRESHOLD = 4.0          # z-score cutoff for spike removal
MIN_SEWERSHED_COVERAGE = 0.5       # drop county-week if <50% sewersheds report
MIN_RECOVERY_EFFICIENCY = 10.0     # percent; samples below this rec_eff_percent are QC-failed
TARGET_UNIT = "copies/g dry sludge"  # primary unit — covers all 9 Bay Area counties
SECONDARY_UNIT = "copies/l wastewater"  # liquid track; used only for Section 4.1 comparison
INTERPOLATION_MAX_GAP = 7          # days; gaps longer than this are left NaN
ROLLING_SMOOTH_DAYS = 7            # smoothing window for flow-weighted signal

# Feature engineering
LAG_FEATURES_DAYS:  list[int] = [7, 14, 21]
ROLLING_WINDOWS:    list[int] = [7, 14, 28]    # days for rolling mean / std features
FOURIER_ORDER = 3                              # annual + semiannual seasonality


# ---------------------------------------------------------------------------
# TFT Model Hyperparameters
# ---------------------------------------------------------------------------

# Forecast horizon & context
# H = 8 weeks: clinically meaningful alert window (public-health informed).
# INPUT_SIZE = 26 weeks: one epidemiological half-year of lookback (≥ 3 × H).
H = 8               # forecast horizon (weekly steps)
INPUT_SIZE = 26     # lookback context window (weekly steps); ≥ 3 × H

# Architecture
TFT_CONFIG: dict = {
    "h":                   H,
    "input_size":          INPUT_SIZE,

    # Core dimensions
    "hidden_size":         128,         # d_model — embedding & LSTM hidden dim
    "n_head":              4,           # multi-head self-attention heads
    "attn_dropout":        0.1,
    "dropout":             0.1,
    "ff_hidden_size":      256,         # feed-forward sublayer width (2× hidden_size)

    # Encoder / decoder depth
    "encoder_layers":      2,           # number of LSTM encoder layers
    "decoder_layers":      2,

    # Variable selection networks
    "n_quantiles":         7,           # mirrors QUANTILE_LEVELS length
    "quantile_levels":     [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975],

    # Training
    "max_steps":           2000,
    "learning_rate":       1e-3,
    "batch_size":          64,
    "valid_batch_size":    128,
    # identity: processor RobustScaler already standardises all features.
    # A second NeuralForecast robust pass double-compresses variance, collapsing
    # quantile spread and causing 0% PI coverage.
    "scaler_type":         "identity",
    "early_stop_patience_steps": 100,

    # Pad series shorter than input_size with zeros rather than raising an error.
    # Required because Bay Area counties started NWSS reporting at different times
    # (e.g. Napa has ~2 weekly rows vs San Francisco's ~35 in the initial fold).
    "start_padding_enabled": True,
    # output_activation intentionally omitted: PINNWastewaterLoss.domain_map
    # no longer applies Softplus (target is RobustScaled and can be negative).

    # Near-term steps weighted 2× to force the model to learn the WW→cases lead
    # window (weeks 1–4) rather than fitting the easier long-tail baseline.
    # Length must equal H.
    "horizon_weight":      [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.8, 0.8],
}

# Static covariates fed to TFT variable selection
STATIC_COVARIATES: list[str] = [
    "log_population",           # log of county population
    "sewershed_count",          # number of reporting sewersheds
    "urban_density_index",      # 0–1 normalised census density
]

# Time-varying known future covariates (calendar features)
FUTURE_COVARIATES: list[str] = [
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_holiday",
    "days_since_last_holiday",
]

# Time-varying past-only covariates
PAST_COVARIATES: list[str] = [
    "log1p_concentration_7d_ma",
    "log1p_concentration_14d_ma",
    "log1p_concentration_7d_std",
    "flow_rate_normalized",
    "growth_rate_7d",           # (x_t - x_{t-7}) / x_{t-7}
    "variant_shannon_entropy",  # diversity of circulating variants (from CDC)
]


# ---------------------------------------------------------------------------
# PINN Growth Rate Penalty
# ---------------------------------------------------------------------------

# Biologically plausible max doubling time ~2 days → max daily growth rate ≈ ln(2)/2
# Lambda set to 0.0 (disabled) for Phase 2 calibration: even at 0.005 the penalty
# suppresses the rapid growth trajectories in Folds 1–2 (BQ.1 onset) and the
# holdout (XBB.1.5 peak).  With 0% PI coverage the priority is getting quantile
# spread right; the biological prior can be restored once coverage is healthy.
GROWTH_RATE_LAMBDA = 0.0            # PINN growth-rate penalty disabled (Phase 2 calibration)
MAX_DAILY_GROWTH_RATE = 0.35        # ln(2)/2 ≈ 0.347; weekly threshold = ×7 ≈ 2.45

# Underdispersion penalty — penalises the model when the predicted 95% PI is
# narrower than MIN_PI_WIDTH scaled units.  After RobustScaling the IQR ≈ 1.0,
# so a well-calibrated 95% PI for a normal distribution spans ≈ 2.9 scaled
# units (±1.96σ, σ≈IQR/1.35).  Phase 3 values: lambda=0.5 (5× Phase 2) and
# MIN_PI_WIDTH=2.5 (~86% of theoretical width) — strong enough to overcome the
# pinball gradient that naturally collapses PIs, without dominating training.
UNDERDISPERSION_LAMBDA = 0.5        # weight of underdispersion penalty (Phase 3: 0.1→0.5)
MIN_PI_WIDTH = 2.5                  # minimum 95% PI width in scaled units (Phase 3: 1.5→2.5)

QUANTILE_LEVELS = TFT_CONFIG["quantile_levels"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Weighted Interval Score interval pairs  (lower_q, upper_q, weight)
WIS_INTERVALS: list[tuple[float, float, float]] = [
    (0.025, 0.975, 0.5),
    (0.25,  0.75,  0.5),
]
WIS_MEDIAN_QUANTILE = 0.50

# Outbreak detection threshold: signal > OUTBREAK_PERCENTILE of trailing baseline
OUTBREAK_PERCENTILE = 90           # percentile of 90-day rolling baseline
OUTBREAK_SUSTAINED_DAYS = 3        # must stay above threshold for N days
OUTBREAK_GROWTH_THRESHOLD = 0.25   # 25% WoW increase flags onset (Phase 3: lowered from 0.40 to lift sensitivity for AUC)
LEAD_TIME_WINDOW_MIN = 7           # days before clinical confirmation
LEAD_TIME_WINDOW_MAX = 21


# ---------------------------------------------------------------------------
# Visualization / Dashboard
# ---------------------------------------------------------------------------

DASH_HOST = "0.0.0.0"
DASH_PORT = 8050
DASH_DEBUG = False

ATTENTION_TOP_K = 10               # top-K variable importances to render


# ---------------------------------------------------------------------------
# Logging & Reporting
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)

# ---------------------------------------------------------------------------
# Local LLM (LM Studio — OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

# LM Studio serves a local OpenAI-compatible API.  Set USE_LOCAL_LLM=true in
# .env to enable automated public health bulletins without any cloud API key.
LOCAL_LLM_BASE_URL = "http://127.0.0.1:1234/v1"

# Model identifier as it appears in LM Studio's loaded-model list.
# Override with LOCAL_LLM_MODEL in .env if you load a different checkpoint.
LOCAL_LLM_MODEL = "gemma-4-26b-it"

LLM_MAX_TOKENS = 2048


# ---------------------------------------------------------------------------
# Environment-backed overrides (loaded from .env by main.py)
# ---------------------------------------------------------------------------

class EnvSettings(BaseSettings):
    use_local_llm:     bool = Field(default=False, alias="USE_LOCAL_LLM")
    local_llm_base_url: str = Field(default=LOCAL_LLM_BASE_URL, alias="LOCAL_LLM_BASE_URL")
    local_llm_model:   str  = Field(default=LOCAL_LLM_MODEL,    alias="LOCAL_LLM_MODEL")
    wandb_api_key:     str  = Field(default="",                  alias="WANDB_API_KEY")
    data_cache_dir:    str  = Field(default=str(PROCESSED_DIR),  alias="DATA_CACHE_DIR")

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


# Instantiated lazily in main.py after dotenv load
settings: EnvSettings | None = None
