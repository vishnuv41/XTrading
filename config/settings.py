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
class AppConfig:
    env: str = os.getenv("APP_ENV", "development")  # development | staging | production
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8001"))

    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    symbols: SymbolsConfig = field(default_factory=SymbolsConfig)

    @property
    def is_production(self) -> bool:
        return self.env == "production"


# Single shared instance — import this, don't instantiate AppConfig yourself,
# so every module sees identical config within one process.
settings = AppConfig()
