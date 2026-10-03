"""
config/settings.py
-------------------
Single source of truth for all environment-driven configuration.
Everything else in person1/ (database, exchange, ingestion, api) should
import from here rather than reading os.environ directly, so there's
exactly one place that knows how config is sourced.

Values come from a .env file (via python-dotenv) with env vars taking
precedence, so the same code runs locally and in a container without
changes — just a different .env or different injected env vars.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root if present. Safe to call even if the
# file doesn't exist (e.g. in production where env vars are injected
# directly by the deployment platform).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_list(key: str, default: list) -> list:
    val = os.getenv(key)
    if val is None:
        return default
    return [item.strip() for item in val.split(",") if item.strip()]


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "xtrading")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "")
    # Separate pool sizes for the sync (psycopg2) and async (asyncpg)
    # paths, since ingestion scripts and the API server have different
    # concurrency profiles.
    pool_min_size: int = int(os.getenv("DB_POOL_MIN_SIZE", "2"))
    pool_max_size: int = int(os.getenv("DB_POOL_MAX_SIZE", "10"))

    @property
    def sync_url(self) -> str:
        """postgresql:// URL for psycopg2 / SQLAlchemy sync engine."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def async_url(self) -> str:
        """postgresql:// URL for asyncpg (no driver suffix needed — asyncpg uses this form directly)."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass(frozen=True)
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: str = os.getenv("REDIS_PASSWORD", "") or None

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class ExchangeConfig:
    # Which venue ingestion targets by default; other/multiple exchanges
    # can still be instantiated directly via exchange/<name>.py.
    default_exchange: str = os.getenv("DEFAULT_EXCHANGE", "binance")
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    use_testnet: bool = _env_bool("EXCHANGE_USE_TESTNET", True)
    # ccxt rate-limit is requests/min; keep conservative by default so a
    # misconfigured scheduler can't get the account rate-limited or banned.
    rate_limit_per_min: int = int(os.getenv("EXCHANGE_RATE_LIMIT_PER_MIN", "1200"))


@dataclass(frozen=True)
class SymbolsConfig:
    # Comma-separated in .env, e.g. SYMBOLS=BTC/USDT,ETH/USDT
    symbols: list = field(default_factory=lambda: _env_list("SYMBOLS", ["BTC/USDT"]))
    # Timeframes ingestion maintains per symbol.
    timeframes: list = field(default_factory=lambda: _env_list("TIMEFRAMES", ["1m", "5m", "15m", "1h", "4h", "1d"]))
    # The base timeframe actually pulled from the exchange; others are
    # derived from this one via preprocessing/resampler.py rather than
    # each being fetched independently (fewer API calls, guaranteed
    # consistency between timeframes).
    base_timeframe: str = os.getenv("BASE_TIMEFRAME", "1m")


@dataclass(frozen=True)
class MLConfig:
    # Minimum calibrated confidence to act on a model signal instead of
    # HOLDing. Locked in from sweep_confidence.py backtesting on
    # BTC/USDT 1h (57 trades, 57.9% win rate, +4.37% return, Sharpe 1.22,
    # profit_factor 1.18 net of 10bps/side costs) — do not bump this
    # back down without re-validating against a fresh out-of-sample
    # window first; see sweep_confidence.py.
    confidence_threshold: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.70"))
    use_percentile_gating: bool = os.getenv("USE_PERCENTILE_GATING", "true").lower() == "true"
    percentile_cutoff_pct: float = float(os.getenv("ML_PERCENTILE_CUTOFF_PCT", "1.0"))


@dataclass(frozen=True)
class RiskConfig:
    # Account equity used for position sizing when the caller doesn't
    # pass a live balance explicitly (e.g. local smoke tests).
    account_equity: float = float(os.getenv("RISK_ACCOUNT_EQUITY", "10000"))
    # Base fixed-fractional risk per trade before confidence/vol scaling.
    base_risk_pct: float = float(os.getenv("RISK_BASE_RISK_PCT", "0.01"))
    # ATR multiplier for stop-loss placement. Locked in from the
    # pt_mult=3.0/sl_mult=1.5 sweep-confidence winner on BTC/USDT 1h
    # (+10.47% return, Sharpe 2.89, 64.7% win rate, survives 10bps costs).
    sl_atr_multiplier: float = float(os.getenv("RISK_SL_ATR_MULTIPLIER", "1.5"))
    # Take-profit expressed as a risk multiple (pt_mult / sl_mult = 3.0/1.5).
    tp_risk_reward: float = float(os.getenv("RISK_TP_RISK_REWARD", "2.0"))
    # Trades with computed R:R below this are downgraded to HOLD before
    # they ever reach position sizing (risk_engine/risk_reward.py).
    min_risk_reward: float = float(os.getenv("RISK_MIN_RISK_REWARD", "1.5"))
    # Portfolio-level caps (risk_engine/portfolio_risk.py).
    max_open_trades: int = int(os.getenv("RISK_MAX_OPEN_TRADES", "5"))
    max_portfolio_heat: float = float(os.getenv("RISK_MAX_PORTFOLIO_HEAT", "0.06"))
    max_correlated_positions: int = int(os.getenv("RISK_MAX_CORRELATED_POSITIONS", "2"))
    correlation_threshold: float = float(os.getenv("RISK_CORRELATION_THRESHOLD", "0.7"))
    # Circuit breaker: if today's realized P&L (as a fraction of equity,
    # negative = loss) is at or below -max_daily_loss_pct, no new trades
    # are opened regardless of what the model/strategy says.
    max_daily_loss_pct: float = float(os.getenv("RISK_MAX_DAILY_LOSS_PCT", "0.03"))
    # Consecutive-loss cooldown: after this many losing trades in a row
    # (same symbol), pause new entries for cooldown_bars. Cheap guard
    # against a bad regime/model-drift stretch compounding losses back
    # to back — the daily-loss % breaker alone doesn't catch a losing
    # streak that stays under the daily threshold but repeats for days.
    max_consecutive_losses: int = int(os.getenv("RISK_MAX_CONSECUTIVE_LOSSES", "3"))
    cooldown_bars: int = int(os.getenv("RISK_COOLDOWN_BARS", "12"))
    # Hard cap on new entries opened per rolling 24h window per symbol,
    # independent of confidence/signal quality — bounds worst-case fee/
    # slippage drag and overtrading if the model starts firing rapidly.
    max_trades_per_day: int = int(os.getenv("RISK_MAX_TRADES_PER_DAY", "8"))


@dataclass(frozen=True)
class AppConfig:
    env: str = os.getenv("APP_ENV", "development")  # development | staging | production
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8001"))

    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    symbols: SymbolsConfig = field(default_factory=SymbolsConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    @property
    def is_production(self) -> bool:
        return self.env == "production"


# Single shared instance — import this, don't instantiate AppConfig yourself,
# so every module sees identical config within one process.
settings = AppConfig()