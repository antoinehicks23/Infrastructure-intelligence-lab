"""
=============================================================================
VP-ORB ALGORITHMIC TRADING SYSTEM
Volume Profile Value Area + Opening Range Breakout
Futures & Forex | 3:1 Risk:Reward | $2,000 Starting Capital
=============================================================================

ARCHITECTURE OVERVIEW
---------------------
Layer 1: Market Structure Filter     — Are we in a tradeable regime?
Layer 2: Volume Profile Engine       — Where is the 70% value area (VAH/VAL/POC)?
Layer 3: Opening Range Breakout      — ORB high/low defined at session open
Layer 4: Signal Confluence Engine    — Both layers must agree
Layer 5: Trade Management            — Entry, stop, 3:1 target, trail
Layer 6: Risk Engine                 — Position sizing, daily max loss, drawdown guard
Layer 7: Noise Filter                — ATR volatility gate + volume confirmation
Layer 8: Adaptive Learning Logger    — Log every trade for edge refinement

CAPITAL ALLOCATION (starting $2,000)
--------------------------------------
- Max risk per trade: 1.5% of equity = $30
- Max daily loss: 4% = $80
- Max concurrent positions: 2
- Scale up: every $500 gained, recalculate position sizes

SUPPORTED INSTRUMENTS
- Futures:  ES (S&P 500), NQ (Nasdaq), CL (Crude), GC (Gold)
- Forex:    EURUSD, GBPUSD, USDJPY, AUDUSD

USAGE
-----
pip install pandas numpy ta requests python-dotenv

Run backtest:
    python vp_orb_system.py --mode backtest --symbol ES --start 2024-01-01 --end 2025-01-01

Run live paper:
    python vp_orb_system.py --mode paper --symbol EURUSD
"""

import pandas as pd
import numpy as np
import logging
import json
import os
import argparse
from datetime import datetime, time, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from enum import Enum

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trade_log.txt"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("VP_ORB")


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────


class Direction(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"

class RegimeState(Enum):
    TRENDING_UP   = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING       = "RANGING"
    HIGH_VOL      = "HIGH_VOL"      # noise — avoid

@dataclass
class VolumeProfile:
    """Holds the computed value area for a given session or lookback window."""
    poc:   float = 0.0   # Point of Control — highest volume price
    vah:   float = 0.0   # Value Area High  — upper bound of 70% vol
    val:   float = 0.0   # Value Area Low   — lower bound of 70% vol
    total_volume: float = 0.0

@dataclass
class ORBLevels:
    """Opening Range Breakout levels for the session."""
    high:      float = 0.0
    low:       float = 0.0
    midpoint:  float = 0.0
    range_pts: float = 0.0   # raw range in points/pips
    valid:     bool  = False

@dataclass
class TradeSignal:
    """Fully qualified trade signal — all layers must pass."""
    direction:   Direction = Direction.FLAT
    entry:       float     = 0.0
    stop:        float     = 0.0
    target:      float     = 0.0
    risk_pts:    float     = 0.0
    reward_pts:  float     = 0.0
    rr_ratio:    float     = 0.0
    size:        float     = 0.0     # contracts or lots
    confidence:  float     = 0.0     # 0–1 composite score
    reason:      str       = ""
    timestamp:   str       = ""

@dataclass
class TradeResult:
    """Closed trade record for logging and edge analysis."""
    signal:       TradeSignal = field(default_factory=TradeSignal)
    exit_price:   float  = 0.0
    exit_time:    str    = ""
    pnl_pts:      float  = 0.0
    pnl_dollars:  float  = 0.0
    outcome:      str    = ""        # WIN / LOSS / BE
    hold_bars:    int    = 0
    exit_reason:  str    = ""        # target / stop / trail / time / manual


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

INSTRUMENT_CONFIG = {
    # symbol: (tick_size, tick_value_usd, pip_factor, orb_minutes, session_open_utc)
    # Micro contracts for $5K capital management
    "MCL":    (0.01,   1.00,   1.0,  1440, time(0,  0)),  # Micro Crude Oil - weekly bars
    "MGC":    (0.10,   1.00,   1.0,  1440, time(0,  0)),  # Micro Gold - weekly bars
    "GC":     (0.10,  10.00,   1.0,  1440, time(0,  0)),  # Gold full - weekly bars
    "CL":     (0.01,  10.00,   1.0,  1440, time(0,  0)),  # Crude full - weekly bars
    "EURUSD": (0.0001, 1.00, 10000,  1440, time(0,  0)),  # Weekly bars
    "USDJPY": (0.01,   1.00,  1000,  1440, time(0,  0)),  # Weekly bars
    "AUDUSD": (0.0001, 1.00, 10000,  1440, time(0,  0)),  # Weekly bars
    "HG":     (0.0005, 12.50,  1.0,  1440, time(0,  0)),  # Copper full - weekly bars
}
SYSTEM_CONFIG = {
    # Risk
    "starting_capital":       2000.0,
    "max_risk_pct_per_trade":    0.015,   # 1.5%
    "max_daily_loss_pct":        0.04,    # 4%
    "max_concurrent_positions":  2,
    "target_rr":                 3.0,     # 3:1 minimum

    # Volume Profile
    "value_area_pct":            0.70,    # 70% of volume
    "vp_lookback_bars":          52,     # bars used to build profile
    "price_bucket_pct":          0.001,   # price bucket granularity

    # ORB
    "orb_minutes":               1440,      # configurable per instrument
    "orb_breakout_buffer":       0.0002,  # 2bp buffer above/below ORB

    # Noise filters
    "atr_period":                14,
    "atr_max_multiple":          2.5,     # if spread > 2.5x ATR, skip — high vol noise
    "atr_min_multiple":          0.5,     # if range < 0.5x ATR, skip — no volatility
    "min_volume_percentile":     40,      # bar volume must be above 40th percentile

    # Regime
    "ema_fast":                  8,
    "ema_slow":                  21,
    "ema_trend":                 50,
    "adx_period":                14,
    "adx_trend_threshold":       22,      # ADX > 22 = trending

    # Trade management
    "trail_activate_at_1r":      True,    # move stop to BE after 1R
    "trail_at_2r_to_1r":         True,    # trail stop to +1R after 2R
    "max_hold_bars":             5,      # time-based exit
    "time_cutoff_minutes_before_close": 0,

    # Scaling
    "equity_step_for_rescale":   500.0,   # recalculate size every $500 gain
}


# ─────────────────────────────────────────────
# LAYER 1: MARKET REGIME FILTER
# ─────────────────────────────────────────────

class RegimeFilter:
    """
    Determines the current market regime so we only trade in conditions
    where our edge exists. VP + ORB works best in trending or breakout
    regimes — NOT in choppy, low-ADX consolidation.

    Uses:
    - EMA stack (8/21/50)     — trend direction and alignment
    - ADX (14)                — trend strength vs noise
    - ATR expansion/contraction — volatility state
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def classify(self, df: pd.DataFrame) -> RegimeState:
        """
        Returns the current RegimeState.
        df must have columns: close, high, low, volume
        All indicators computed fresh here to avoid lookahead.
        """
        if len(df) < self.cfg["ema_trend"] + 5:
            return RegimeState.RANGING  # not enough data

        close = df["close"]

        # EMA stack
        ema_f = close.ewm(span=self.cfg["ema_fast"],  adjust=False).mean()
        ema_s = close.ewm(span=self.cfg["ema_slow"],  adjust=False).mean()
        ema_t = close.ewm(span=self.cfg["ema_trend"], adjust=False).mean()

        # ADX via Wilder's method
        adx = self._compute_adx(df)

        # ATR
        atr = self._compute_atr(df)
        current_atr  = atr.iloc[-1]
        atr_ma        = atr.rolling(20).mean().iloc[-1]

        trending      = adx.iloc[-1] > self.cfg["adx_trend_threshold"]
        bull_stack    = ema_f.iloc[-1] > ema_s.iloc[-1] > ema_t.iloc[-1]
        bear_stack    = ema_f.iloc[-1] < ema_s.iloc[-1] < ema_t.iloc[-1]
        vol_expanding = current_atr > atr_ma * self.cfg["atr_max_multiple"]

        if vol_expanding:
            return RegimeState.HIGH_VOL   # too chaotic — noise dominates

        if trending and bull_stack:
            return RegimeState.TRENDING_UP
        if trending and bear_stack:
            return RegimeState.TRENDING_DOWN

        return RegimeState.RANGING

    def _compute_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        p = period or self.cfg["atr_period"]
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/p, adjust=False).mean()

    def _compute_adx(self, df: pd.DataFrame) -> pd.Series:
        p    = self.cfg["adx_period"]
        high = df["high"]
        low  = df["low"]
        close= df["close"]

        up_move   = high.diff()
        down_move = -low.diff()

        plus_dm  = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index
        )

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)

        atr      = tr.ewm(alpha=1/p, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(alpha=1/p,  adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/p, adjust=False).mean() / atr

        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.ewm(alpha=1/p, adjust=False).mean()
        return adx


# ─────────────────────────────────────────────
# LAYER 2: VOLUME PROFILE ENGINE
# ─────────────────────────────────────────────

class VolumeProfileEngine:
    """
    Builds a classic TPO/volume distribution over a lookback window,
    then identifies POC, VAH, and VAL.

    The 70% Value Area rule:
    - Start at the POC (peak volume node).
    - Expand up and down, adding nodes by highest volume first,
      until the cumulative volume >= 70% of total session volume.
    - VAH = highest price bucket included.
    - VAL = lowest price bucket included.

    Why this matters:
    - Price inside the VA = "accepted" — market is balanced.
    - Price outside VA = "rejected" or "exploring" — directional move likely.
    - Breakout from VA with ORB confirmation = high-probability entry.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def build(self, df: pd.DataFrame) -> VolumeProfile:
        """
        Computes the volume profile from the last N bars.
        df must have: high, low, close, volume
        """
        if len(df) < 20:
            return VolumeProfile()

        lookback = df.tail(self.cfg["vp_lookback_bars"]).copy()
        total_vol = lookback["volume"].sum()
        if total_vol == 0:
            return VolumeProfile()

        # Determine price range and bucket size
        price_min = lookback["low"].min()
        price_max = lookback["high"].max()
        bucket_size = (price_max - price_min) * self.cfg["price_bucket_pct"]
        if bucket_size <= 0:
            bucket_size = 0.01

        # Build volume-at-price distribution
        # Each bar's volume is distributed proportionally across its high-low range
        buckets: Dict[float, float] = {}
        for _, row in lookback.iterrows():
            bar_range = row["high"] - row["low"]
            if bar_range <= 0:
                key = round(row["close"] / bucket_size) * bucket_size
                buckets[key] = buckets.get(key, 0) + row["volume"]
                continue

            lo_key = int(row["low"]  / bucket_size)
            hi_key = int(row["high"] / bucket_size) + 1
            n_buckets = hi_key - lo_key or 1

            for b in range(lo_key, hi_key):
                price_key = round(b * bucket_size, 6)
                share = row["volume"] / n_buckets
                buckets[price_key] = buckets.get(price_key, 0) + share

        if not buckets:
            return VolumeProfile()

        # Sort by volume to find POC
        sorted_by_vol  = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
        poc_price, poc_vol = sorted_by_vol[0]

        # Expand value area from POC until >= 70%
        target_vol = total_vol * self.cfg["value_area_pct"]
        included   = {poc_price: poc_vol}
        cum_vol    = poc_vol

        sorted_prices = sorted(buckets.keys())
        poc_idx       = sorted_prices.index(poc_price)
        lo_idx        = poc_idx
        hi_idx        = poc_idx

        while cum_vol < target_vol:
            # Candidate nodes: next above and next below
            can_go_up   = hi_idx < len(sorted_prices) - 1
            can_go_down = lo_idx > 0

            if not can_go_up and not can_go_down:
                break

            # Choose the side with higher volume node to add
            up_vol   = buckets.get(sorted_prices[hi_idx + 1], 0) if can_go_up   else -1
            down_vol = buckets.get(sorted_prices[lo_idx - 1], 0) if can_go_down else -1

            if up_vol >= down_vol:
                hi_idx += 1
                p = sorted_prices[hi_idx]
            else:
                lo_idx -= 1
                p = sorted_prices[lo_idx]

            included[p] = buckets[p]
            cum_vol += buckets[p]

        vah = max(included.keys())
        val = min(included.keys())

        return VolumeProfile(
            poc=poc_price,
            vah=vah,
            val=val,
            total_volume=total_vol
        )

    def price_in_value_area(self, price: float, vp: VolumeProfile) -> bool:
        return vp.val <= price <= vp.vah

    def price_above_value_area(self, price: float, vp: VolumeProfile) -> bool:
        return price > vp.vah

    def price_below_value_area(self, price: float, vp: VolumeProfile) -> bool:
        return price < vp.val


# ─────────────────────────────────────────────
# LAYER 3: OPENING RANGE BREAKOUT (ORB)
# ─────────────────────────────────────────────

class ORBEngine:
    """
    Defines the Opening Range (OR) as the high/low formed during the
    first N minutes of the session.

    ORB Signal:
    - LONG:  price breaks and closes above OR high
    - SHORT: price breaks and closes below OR low

    ORB is only valid when:
    1. Range is between 0.5x and 2.5x of the instrument's ATR
       (too narrow = nothing happening; too wide = already extended)
    2. Volume during the ORB period is above the rolling median

    Combined with Volume Profile:
    - Best ORB LONG:  ORB high sits at or above VAH (confirming breakout from VA)
    - Best ORB SHORT: ORB low sits at or below VAL
    - Second-tier:    ORB break toward POC from outside VA (mean reversion mode)
    """

    def __init__(self, cfg: dict, instrument: str):
        self.cfg        = cfg
        self.instrument = instrument
        tick_sz, tick_val, pip_f, orb_min, sess_open = INSTRUMENT_CONFIG[instrument]
        self.orb_minutes = orb_min
        self.session_open = sess_open

    def build_orb(self, df: pd.DataFrame) -> ORBLevels:
        """
        df: intraday OHLCV with DatetimeIndex (UTC).
        Returns ORB levels for the most recent session.
        """
        if df.index.tzinfo is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")

        today = df.index[-1].date()
        session_start = pd.Timestamp(
            datetime.combine(today, self.session_open), tz="UTC"
        )
        session_end   = session_start + timedelta(minutes=self.orb_minutes)

        orb_bars = df[(df.index >= session_start) & (df.index < session_end)]

        if len(orb_bars) < 2:
            return ORBLevels(valid=False)

        orb_high = orb_bars["high"].max()
        orb_low  = orb_bars["low"].min()
        orb_mid  = (orb_high + orb_low) / 2
        orb_range = orb_high - orb_low

        return ORBLevels(
            high      = orb_high,
            low       = orb_low,
            midpoint  = orb_mid,
            range_pts = orb_range,
            valid     = True
        )

    def is_orb_breakout(
        self, current_bar: pd.Series, orb: ORBLevels
    ) -> Tuple[bool, Direction]:
        """
        Returns (is_breakout, direction).
        Requires a close ABOVE ORB high or BELOW ORB low, not just a wick.
        """
        buf = self.cfg["orb_breakout_buffer"]
        if not orb.valid:
            return False, Direction.FLAT

        if current_bar["close"] > orb.high * (1 + buf):
            return True, Direction.LONG
        if current_bar["close"] < orb.low  * (1 - buf):
            return True, Direction.SHORT

        return False, Direction.FLAT


# ─────────────────────────────────────────────
# LAYER 4: NOISE FILTER
# ─────────────────────────────────────────────

class NoiseFilter:
    """
    Separates signal from noise by checking three independent dimensions:

    1. ATR Volatility Gate
       - If current bar range < 0.5x ATR → dead market, skip
       - If current bar range > 2.5x ATR → chaotic spike, skip
       - The "sweet spot" is a bar with meaningful range but not excessive

    2. Volume Confirmation
       - The breakout bar's volume must be above the rolling 40th percentile
       - Low-volume breakouts are false breakouts ~60-70% of the time
       - We want conviction from market participants

    3. Spread / Slippage Check (for forex)
       - If live bid/ask spread > 1.5x typical spread, skip — cost kills edge

    Noise vs Signal heuristics (embedded in code):
    - Signal: high-volume bar breaking a key level (VAH/VAL/ORB) after consolidation
    - Noise: random bar breaking a level on thin volume mid-session
    - Signal: ORB break that aligns WITH the trend (regime filter agrees)
    - Noise: ORB break AGAINST the trend regime
    - Signal: breakout bar has clean close, minimal upper/lower wick
    - Noise: breakout bar has long opposing wick (price rejected)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def passes(
        self,
        df:          pd.DataFrame,
        signal_bar:  pd.Series,
        direction:   Direction
    ) -> Tuple[bool, str]:
        """
        Returns (passes, reason_if_failed).
        """
        atr      = self._atr(df).iloc[-1]
        bar_range = signal_bar["high"] - signal_bar["low"]

        # 1. ATR volatility gate
        if bar_range < atr * self.cfg["atr_min_multiple"]:
            return False, f"Bar range {bar_range:.5f} < ATR min threshold {atr * self.cfg['atr_min_multiple']:.5f}"
        if bar_range > atr * self.cfg["atr_max_multiple"]:
            return False, f"Bar range {bar_range:.5f} > ATR max threshold — spike/noise"

        # 2. Volume confirmation
        vol_percentile = df["volume"].rolling(50).quantile(
            self.cfg["min_volume_percentile"] / 100
        ).iloc[-1]
        if signal_bar["volume"] < vol_percentile:
            return False, f"Volume {signal_bar['volume']:.0f} below {self.cfg['min_volume_percentile']}th percentile"

        # 3. Wick rejection filter
        # A long opposing wick means smart money rejected the breakout direction
        body  = abs(signal_bar["close"] - signal_bar["open"])
        if direction == Direction.LONG:
            upper_wick = signal_bar["high"]  - max(signal_bar["close"], signal_bar["open"])
            if upper_wick > body * 1.5 and body > 0:
                return False, "Long upper wick on breakout bar — possible rejection"
        else:
            lower_wick = min(signal_bar["close"], signal_bar["open"]) - signal_bar["low"]
            if lower_wick > body * 1.5 and body > 0:
                return False, "Long lower wick on breakout bar — possible rejection"

        return True, ""

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        p  = self.cfg["atr_period"]
        h, l, c = df["high"], df["low"], df["close"]
        pc = c.shift(1)
        tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/p, adjust=False).mean()


# ─────────────────────────────────────────────
# LAYER 5: SIGNAL CONFLUENCE ENGINE
# ─────────────────────────────────────────────

class SignalEngine:
    """
    The decision layer. Requires ALL of the following to align
    before a trade is generated:

    LONG confluence:
    ✓ Regime = TRENDING_UP or RANGING (not HIGH_VOL)
    ✓ ORB breakout to the upside confirmed
    ✓ ORB high is near or above VAH (breakout from Value Area)
    ✓ Noise filter passes
    ✓ Risk:Reward >= 3:1 with natural stop placement

    SHORT confluence:
    ✓ Regime = TRENDING_DOWN or RANGING
    ✓ ORB breakout to the downside confirmed
    ✓ ORB low is near or below VAL
    ✓ Noise filter passes
    ✓ R:R >= 3:1

    BACKING OUT OF A TRADE (invalidation after entry):
    - Stop is hit (hard rule, never moved against)
    - Price re-enters Value Area after breakout (failed breakout)
    - Regime flips to HIGH_VOL
    - Time cutoff reached (30 min before session close)
    """

    def __init__(self, cfg: dict, instrument: str):
        self.cfg        = cfg
        self.instrument = instrument
        tick_sz, tick_val, pip_f, orb_min, sess_open = INSTRUMENT_CONFIG[instrument]
        self.tick_size  = tick_sz
        self.tick_value = tick_val

    def evaluate(
        self,
        df:     pd.DataFrame,
        vp:     VolumeProfile,
        orb:    ORBLevels,
        regime: RegimeState,
        equity: float
    ) -> Optional[TradeSignal]:
        """
        Main signal generation function.
        Returns a TradeSignal if all conditions met, else None.
        """
        if regime == RegimeState.HIGH_VOL:
            log.debug("Regime HIGH_VOL — no trades")
            return None

        if not orb.valid:
            log.debug("ORB not valid yet")
            return None

        current_bar = df.iloc[-1]
        atr = self._atr(df).iloc[-1]

        # Regime direction filter
        regime_allows_long  = regime in [RegimeState.TRENDING_UP,   RegimeState.RANGING]
        regime_allows_short = regime in [RegimeState.TRENDING_DOWN,  RegimeState.RANGING]

        # ORB breakout detection
        noise = NoiseFilter(self.cfg)
        orb_eng = ORBEngine(self.cfg, self.instrument)
        is_break, direction = orb_eng.is_orb_breakout(current_bar, orb)

        if not is_break or direction == Direction.FLAT:
            return None

        # Regime gating
        if direction == Direction.LONG  and not regime_allows_long:
            log.debug("LONG signal blocked by TRENDING_DOWN regime")
            return None
        if direction == Direction.SHORT and not regime_allows_short:
            log.debug("SHORT signal blocked by TRENDING_UP regime")
            return None

        # Volume Profile confluence
        vp_confirmation, vp_reason, confidence_boost = self._vp_confirm(
            direction, orb, vp, current_bar
        )
        if not vp_confirmation:
            log.debug(f"VP filter blocked: {vp_reason}")
            return None

        # Noise filter
        noise_ok, noise_reason = noise.passes(df, current_bar, direction)
        if not noise_ok:
            log.debug(f"Noise filter blocked: {noise_reason}")
            return None

        # Entry, stop, target
        entry, stop, target = self._levels(direction, current_bar, vp, orb, atr)
        risk_pts   = abs(entry - stop)
        reward_pts = abs(target - entry)

        if risk_pts <= 0:
            return None

        rr = reward_pts / risk_pts
        if rr < self.cfg["target_rr"]:
            log.debug(f"R:R {rr:.2f} < minimum {self.cfg['target_rr']}")
            return None

        # Position sizing
        size = self._position_size(equity, risk_pts)

        # Confidence score (0–1) — for logging / future ML input
        confidence = self._confidence_score(regime, direction, confidence_boost)

        signal = TradeSignal(
            direction  = direction,
            entry      = round(entry, 5),
            stop       = round(stop,  5),
            target     = round(target, 5),
            risk_pts   = round(risk_pts,   5),
            reward_pts = round(reward_pts, 5),
            rr_ratio   = round(rr, 2),
            size       = size,
            confidence = round(confidence, 3),
            reason     = f"ORB {direction.value} | VP: {vp_reason} | Regime: {regime.value}",
            timestamp  = str(df.index[-1])
        )
        log.info(f"SIGNAL GENERATED: {signal}")
        return signal

    def _vp_confirm(
        self,
        direction: Direction,
        orb:       ORBLevels,
        vp:        VolumeProfile,
        bar:       pd.Series
    ) -> Tuple[bool, str, float]:
        """
        Returns (confirmed, reason, confidence_boost).

        Tier 1 (highest edge): Price breaks OUT of Value Area in breakout direction
        Tier 2:                 Price holds above/below POC, pullback into VA edge
        Tier 3 (mean rev):      Price extends far from VA and we fade back to POC
        """
        price = bar["close"]

        if direction == Direction.LONG:
            if price > vp.vah:
                return True, f"Above VAH ({vp.vah:.5f}) — breakout", 0.2
            if price > vp.poc and price < vp.vah:
                return True, f"Above POC ({vp.poc:.5f}), approaching VAH", 0.0
            return False, f"Price {price:.5f} below POC {vp.poc:.5f} — no long edge", 0.0

        else:  # SHORT
            if price < vp.val:
                return True, f"Below VAL ({vp.val:.5f}) — breakout", 0.2
            if price < vp.poc and price > vp.val:
                return True, f"Below POC ({vp.poc:.5f}), approaching VAL", 0.0
            return False, f"Price {price:.5f} above POC {vp.poc:.5f} — no short edge", 0.0

    def _levels(
        self,
        direction: Direction,
        bar:       pd.Series,
        vp:        VolumeProfile,
        orb:       ORBLevels,
        atr:       float
    ) -> Tuple[float, float, float]:
        """
        Computes entry/stop/target.

        LONG:
          Entry  = current close (market order on confirmed close)
          Stop   = below ORB low or VAL, whichever is lower — we need both levels
                   to break before calling it invalid
          Target = entry + 3 * risk

        SHORT:
          Entry  = current close
          Stop   = above ORB high or VAH
          Target = entry - 3 * risk
        """
        entry = bar["close"]
        buf   = atr * 0.25   # small buffer below key level to avoid noise stop-out

        if direction == Direction.LONG:
            stop_level = min(orb.low, vp.val) - buf
            stop       = max(stop_level, entry - atr * 1.5)  # cap stop at 1.5x ATR
            target     = entry + self.cfg["target_rr"] * (entry - stop)

        else:  # SHORT
            stop_level = max(orb.high, vp.vah) + buf
            stop       = min(stop_level, entry + atr * 1.5)
            target     = entry - self.cfg["target_rr"] * (stop - entry)

        return entry, stop, target

    def _position_size(self, equity: float, risk_pts: float) -> float:
        """
        Risk-based position sizing.
        dollars_at_risk = equity * max_risk_pct
        size = dollars_at_risk / (risk_pts * tick_value / tick_size)
        """
        max_risk_dollars = equity * self.cfg["max_risk_pct_per_trade"]
        dollar_per_pt    = self.tick_value / self.tick_size
        if risk_pts <= 0 or dollar_per_pt <= 0:
            return 0.0
        size = max_risk_dollars / (risk_pts * dollar_per_pt)
        return max(round(size, 2), 0.01)  # minimum 0.01 lot

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        p  = self.cfg["atr_period"]
        h, l, c = df["high"], df["low"], df["close"]
        pc = c.shift(1)
        tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/p, adjust=False).mean()

    def _confidence_score(
        self,
        regime:    RegimeState,
        direction: Direction,
        boost:     float
    ) -> float:
        base = 0.5
        # Regime alignment adds confidence
        if regime == RegimeState.TRENDING_UP   and direction == Direction.LONG:  base += 0.2
        if regime == RegimeState.TRENDING_DOWN and direction == Direction.SHORT: base += 0.2
        base += boost
        return min(base, 1.0)


# ─────────────────────────────────────────────
# LAYER 6: TRADE MANAGER
# ─────────────────────────────────────────────

class TradeManager:
    """
    Manages open positions in real time.

    Trail stop logic:
    1. On entry: hard stop placed.
    2. After price moves 1R in our favor: move stop to break-even.
    3. After price moves 2R in our favor: move stop to lock in 1R profit.
    4. Target hit or stop hit → close position.

    Invalidation logic (exit before target/stop):
    - Price re-enters Value Area after breakout = failed breakout → EXIT
    - Regime flips to HIGH_VOL → EXIT
    - Time cutoff (30 min before session close) → EXIT at market
    - Max hold bars reached → EXIT

    This is the discipline layer. Do not override stops or extend targets.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.open_trades: Dict[str, dict] = {}

    def open(self, signal: TradeSignal) -> str:
        trade_id = f"T{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        self.open_trades[trade_id] = {
            "signal":      signal,
            "current_stop": signal.stop,
            "bars_held":   0,
            "phase":       0,   # 0=initial, 1=BE, 2=1R-locked
        }
        log.info(f"[{trade_id}] OPENED {signal.direction.value} @ {signal.entry} | Stop {signal.stop} | Target {signal.target}")
        return trade_id

    def update(
        self,
        trade_id:   str,
        current_bar: pd.Series,
        vp:          VolumeProfile,
        regime:      RegimeState,
        session_end: Optional[datetime] = None
    ) -> Tuple[bool, str, float]:
        """
        Returns (should_close, reason, exit_price).
        Called on each new bar while position is open.
        """
        if trade_id not in self.open_trades:
            return False, "", 0.0

        t       = self.open_trades[trade_id]
        signal  = t["signal"]
        price   = current_bar["close"]
        t["bars_held"] += 1

        # Time cutoff
        if t["bars_held"] >= self.cfg["max_hold_bars"]:
            return True, "time_exit_max_bars", price

        # Regime flip to HIGH_VOL
        if regime == RegimeState.HIGH_VOL:
            return True, "regime_high_vol_exit", price

        # Re-entry into Value Area (failed breakout invalidation)
        if signal.direction == Direction.LONG and price < vp.vah:
            if price < vp.poc:
                return True, "price_back_in_value_area", price
        if signal.direction == Direction.SHORT and price > vp.val:
            if price > vp.poc:
                return True, "price_back_in_value_area", price

        # Stop hit
        if signal.direction == Direction.LONG  and price <= t["current_stop"]:
            return True, "stop_hit", t["current_stop"]
        if signal.direction == Direction.SHORT and price >= t["current_stop"]:
            return True, "stop_hit", t["current_stop"]

        # Target hit
        if signal.direction == Direction.LONG  and price >= signal.target:
            return True, "target_hit", signal.target
        if signal.direction == Direction.SHORT and price <= signal.target:
            return True, "target_hit", signal.target

        # Trailing stop adjustments
        r = signal.risk_pts
        if signal.direction == Direction.LONG:
            move = price - signal.entry
            if move >= 2 * r and t["phase"] < 2 and self.cfg["trail_at_2r_to_1r"]:
                t["current_stop"] = signal.entry + r
                t["phase"] = 2
                log.info(f"[{trade_id}] Trail stop → +1R ({t['current_stop']:.5f})")
            elif move >= r and t["phase"] < 1 and self.cfg["trail_activate_at_1r"]:
                t["current_stop"] = signal.entry
                t["phase"] = 1
                log.info(f"[{trade_id}] Stop moved to BE ({signal.entry:.5f})")
        else:
            move = signal.entry - price
            if move >= 2 * r and t["phase"] < 2 and self.cfg["trail_at_2r_to_1r"]:
                t["current_stop"] = signal.entry - r
                t["phase"] = 2
                log.info(f"[{trade_id}] Trail stop → +1R ({t['current_stop']:.5f})")
            elif move >= r and t["phase"] < 1 and self.cfg["trail_activate_at_1r"]:
                t["current_stop"] = signal.entry
                t["phase"] = 1
                log.info(f"[{trade_id}] Stop moved to BE ({signal.entry:.5f})")

        return False, "", 0.0

    def close(self, trade_id: str, exit_price: float, reason: str) -> TradeResult:
        t      = self.open_trades.pop(trade_id)
        signal = t["signal"]
        tick_sz, tick_val, _, _, _ = INSTRUMENT_CONFIG.get(signal.reason.split("|")[0].strip(), (1, 1, 1, 15, time(9, 30)))

        if signal.direction == Direction.LONG:
            pnl_pts = exit_price - signal.entry
        else:
            pnl_pts = signal.entry - exit_price

        # Rough PnL estimate (exact for futures, approximate for forex)
        commission = 4.50 # per round trip
        pnl_dollars = (pnl_pts * (tick_val / tick_sz) * signal.size) - commision

        if pnl_pts > 0:
            outcome = "WIN"
        elif pnl_pts < 0:
            outcome = "LOSS"
        else:
            outcome = "BE"

        result = TradeResult(
            signal       = signal,
            exit_price   = round(exit_price, 5),
            exit_time    = str(datetime.utcnow()),
            pnl_pts      = round(pnl_pts, 5),
            pnl_dollars  = round(pnl_dollars, 2),
            outcome      = outcome,
            hold_bars    = t["bars_held"],
            exit_reason  = reason
        )
        log.info(f"[{trade_id}] CLOSED {outcome} | PnL: {pnl_dollars:.2f} | Reason: {reason}")
        return result


# ─────────────────────────────────────────────
# LAYER 7: RISK ENGINE
# ─────────────────────────────────────────────

class RiskEngine:
    """
    Portfolio-level guardrails. Enforces discipline at the account level.

    Rules:
    1. Max concurrent positions: 2 (avoids overexposure)
    2. Daily max loss: 4% of account. Once hit, system shuts down for the day.
    3. Per-trade max risk: 1.5% of current equity
    4. Scale-up trigger: every $500 gain, recalculate position sizes
       (this is how $2K grows systematically — compounding through sizing)
    5. Drawdown guard: if account drops 15% from peak, reduce risk to 0.75%/trade
       until equity recovers to within 10% of peak.
    """

    def __init__(self, cfg: dict, starting_equity: float):
        self.cfg            = cfg
        self.equity         = starting_equity
        self.equity_peak    = starting_equity
        self.daily_pnl      = 0.0
        self.daily_trades   = 0
        self.trading_halted = False
        self.reduced_risk   = False
        self.trade_history: List[TradeResult] = []

    def can_trade(self, open_positions: int) -> Tuple[bool, str]:
        if self.trading_halted:
            return False, "Daily loss limit reached — trading halted"
        if open_positions >= self.cfg["max_concurrent_positions"]:
            return False, f"Max concurrent positions ({self.cfg['max_concurrent_positions']}) reached"
        return True, ""

    def record_trade(self, result: TradeResult):
        self.trade_history.append(result)
        self.equity     += result.pnl_dollars
        self.daily_pnl  += result.pnl_dollars
        self.daily_trades += 1

        # Update peak
        if self.equity > self.equity_peak:
            self.equity_peak = self.equity

        # Check daily loss limit
        daily_loss_limit = self.cfg["starting_capital"] * self.cfg["max_daily_loss_pct"]
        if self.daily_pnl < -daily_loss_limit:
            self.trading_halted = True
            log.warning(f"DAILY LOSS LIMIT HIT: {self.daily_pnl:.2f}. Trading halted.")

        # Check drawdown guard
        drawdown_pct = (self.equity_peak - self.equity) / self.equity_peak
        if drawdown_pct > 0.15:
            self.reduced_risk = True
            log.warning(f"Drawdown {drawdown_pct:.1%} > 15% — risk reduced to 0.75%")
        elif drawdown_pct < 0.10 and self.reduced_risk:
            self.reduced_risk = False
            log.info("Drawdown recovered — normal risk restored")

    def current_risk_pct(self) -> float:
        if self.reduced_risk:
            return self.cfg["max_risk_pct_per_trade"] * 0.5   # half risk during drawdown
        return self.cfg["max_risk_pct_per_trade"]

    def reset_daily(self):
        self.daily_pnl      = 0.0
        self.daily_trades   = 0
        self.trading_halted = False
        log.info(f"Daily reset. Current equity: {self.equity:.2f}")

    def stats(self) -> dict:
        if not self.trade_history:
            return {}
        wins  = [t for t in self.trade_history if t.outcome == "WIN"]
        losses= [t for t in self.trade_history if t.outcome == "LOSS"]
        pnls  = [t.pnl_dollars for t in self.trade_history]
        return {
            "total_trades":    len(self.trade_history),
            "win_rate":        len(wins) / len(self.trade_history),
            "avg_win":         np.mean([t.pnl_dollars for t in wins])  if wins   else 0,
            "avg_loss":        np.mean([t.pnl_dollars for t in losses]) if losses else 0,
            "profit_factor":   sum(t.pnl_dollars for t in wins) / abs(sum(t.pnl_dollars for t in losses) + 1e-9),
            "total_pnl":       sum(pnls),
            "current_equity":  self.equity,
            "equity_peak":     self.equity_peak,
            "max_drawdown":    (self.equity_peak - min([self.equity_peak - sum(pnls[:i]) for i in range(len(pnls)+1)])) / self.equity_peak
        }
"sharpe ratio": round((np.mean(pnls) / (np.std(pnls) + 1e-9)) * np.sqrt(52), 4) 

# ─────────────────────────────────────────────
# LAYER 8: ADAPTIVE LEARNING LOGGER
# ─────────────────────────────────────────────

class AdaptiveLearningLogger:
    """
    Logs every trade with full context so the edge can be reviewed, measured,
    and improved over time. This is the growth mechanism.

    Each log entry captures:
    - Signal conditions (regime, VP levels, ORB levels, confidence score)
    - Trade outcome (pnl, bars held, exit reason)
    - Market context (ATR, volume, spread)

    The output is a JSONL file that can be loaded into pandas for analysis:
      df = pd.read_json("trade_journal.jsonl", lines=True)
      df.groupby("regime")["pnl_dollars"].mean()
      df[df.outcome == "WIN"]["confidence"].hist()

    Periodic review questions (build into monthly review habit):
    1. Which regimes produce the highest win rate?
    2. Are ORB + VP Tier 1 confluences outperforming Tier 2?
    3. Do we win more when confidence > 0.7?
    4. Is noise filter reducing the loss rate or removing winners?
    5. Which instruments have the best profit factor?
    """

    def __init__(self, filepath: str = "trade_journal.jsonl"):
        self.filepath = filepath

    def log(self, result: TradeResult, extra_context: dict = None):
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [serialize(i) for i in obj]
            return obj
        entry = serialize(asdict(result))
        if extra_context:
            entry.update(extra_context)
        entry["log_time"] = datetime.utcnow().isoformat()

        with open(self.filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_journal(self) -> pd.DataFrame:
        try:
            return pd.read_json(self.filepath, lines=True)
        except Exception:
            return pd.DataFrame()

    def edge_report(self) -> dict:
        df = self.load_journal()
        if df.empty:
            return {"error": "No trades logged yet."}

        report = {}
        for regime in df.get("regime", pd.Series()).unique():
            subset = df[df["regime"] == regime]
            report[regime] = {
                "n":          len(subset),
                "win_rate":   (subset["outcome"] == "WIN").mean(),
                "avg_pnl":    subset["pnl_dollars"].mean()
            }
        return report


# ─────────────────────────────────────────────
# MAIN SYSTEM ORCHESTRATOR
# ─────────────────────────────────────────────

class VPORBSystem:
    """
    Top-level orchestrator.
    Connects all layers. Can run in backtest or live paper mode.

    BACKTEST MODE: iterate over historical OHLCV, generate signals,
    simulate fills at close, track equity curve.

    PAPER MODE: connect to broker/data feed, run signal engine on
    each new bar, submit paper orders.

    LIVE MODE: Same as paper but with live order submission (requires
    broker API integration — not included here to prevent accidental
    live trading).
    """

    def __init__(self, instrument: str, equity: float = None):
        self.instrument = instrument
        self.cfg        = SYSTEM_CONFIG
        self.equity     = equity or self.cfg["starting_capital"]

        self.regime_filter = RegimeFilter(self.cfg)
        self.vp_engine     = VolumeProfileEngine(self.cfg)
        self.orb_engine    = ORBEngine(self.cfg, instrument)
        self.signal_engine = SignalEngine(self.cfg, instrument)
        self.trade_mgr     = TradeManager(self.cfg)
        self.risk_engine   = RiskEngine(self.cfg, self.equity)
        self.logger        = AdaptiveLearningLogger()

        self.current_orb:  Optional[ORBLevels]     = None
        self.current_vp:   Optional[VolumeProfile]  = None
        self.current_regime: RegimeState             = RegimeState.RANGING
        self.open_trade_ids: List[str]               = []

    def run_backtest(self, df: pd.DataFrame) -> dict:
        """
        Run the full system over historical data.
        df: OHLCV DataFrame with DatetimeIndex.
        Returns performance statistics.
        """
        log.info(f"Starting backtest on {self.instrument} | {len(df)} bars")

        # Reset state
        self.open_trade_ids = []
        self.risk_engine.reset_daily()

        for i in range(100, len(df)):
            window = df.iloc[:i+1]
            bar    = df.iloc[i]

            # Day rollover — reset daily state and rebuild ORB
            if i > 100:
                prev_bar = df.iloc[i-1]
                if bar.name.date() != prev_bar.name.date():
                    self.risk_engine.reset_daily()
                    self.current_orb = None

            # Rebuild indicators on each bar (no lookahead)
            self.current_regime = self.regime_filter.classify(window)
            self.current_vp     = self.vp_engine.build(window)

            if self.current_orb is None or not self.current_orb.valid:
                self.current_orb = self.orb_engine.build_orb(window)

            # Manage existing positions
            closed_ids = []
            for tid in self.open_trade_ids:
                should_close, reason, exit_price = self.trade_mgr.update(
                    tid, bar, self.current_vp, self.current_regime
                )
                if should_close:
                    result = self.trade_mgr.close(tid, exit_price, reason)
                    self.risk_engine.record_trade(result)
                    self.logger.log(result, {"regime": self.current_regime.value, "bar_idx": i})
                    closed_ids.append(tid)

            self.open_trade_ids = [t for t in self.open_trade_ids if t not in closed_ids]

            # Check if we can open new trade
            can_trade, block_reason = self.risk_engine.can_trade(len(self.open_trade_ids))
            if not can_trade:
                continue

            # Generate signal
            signal = self.signal_engine.evaluate(
                window,
                self.current_vp,
                self.current_orb,
                self.current_regime,
                self.risk_engine.equity
            )

            if signal:
                tid = self.trade_mgr.open(signal)
                self.open_trade_ids.append(tid)

        # Close any remaining open positions at end of data
        for tid in self.open_trade_ids:
            result = self.trade_mgr.close(tid, df.iloc[-1]["close"], "end_of_data")
            self.risk_engine.record_trade(result)
            self.logger.log(result, {"regime": self.current_regime.value})

        stats = self.risk_engine.stats()
        log.info(f"\n{'='*50}\nBACKTEST COMPLETE\n{json.dumps(stats, indent=2)}\n{'='*50}")
        return stats

    def on_bar(self, df: pd.DataFrame):
        """
        Called on each new bar in live/paper mode.
        df: rolling window of OHLCV data.
        """
        if len(df) < 100:
            return

        bar = df.iloc[-1]

        # Day rollover
        if len(df) > 1 and bar.name.date() != df.iloc[-2].name.date():
            self.risk_engine.reset_daily()
            self.current_orb = None

        # Rebuild on each bar
        self.current_regime = self.regime_filter.classify(df)
        self.current_vp     = self.vp_engine.build(df)
        if self.current_orb is None:
            self.current_orb = self.orb_engine.build_orb(df)

        # Manage positions
        closed_ids = []
        for tid in self.open_trade_ids:
            should_close, reason, exit_price = self.trade_mgr.update(
                tid, bar, self.current_vp, self.current_regime
            )
            if should_close:
                result = self.trade_mgr.close(tid, exit_price, reason)
                self.risk_engine.record_trade(result)
                self.logger.log(result, {"regime": self.current_regime.value})
                closed_ids.append(tid)

        self.open_trade_ids = [t for t in self.open_trade_ids if t not in closed_ids]

        # New signal check
        can_trade, _ = self.risk_engine.can_trade(len(self.open_trade_ids))
        if not can_trade:
            return

        signal = self.signal_engine.evaluate(
            df,
            self.current_vp,
            self.current_orb,
            self.current_regime,
            self.risk_engine.equity
        )

        if signal:
            tid = self.trade_mgr.open(signal)
            self.open_trade_ids.append(tid)
            # ─────────────────────────────────────────────────
            # In live mode: submit order to broker API here
            # Example: broker.submit_order(signal)
            # ─────────────────────────────────────────────────


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def generate_synthetic_ohlcv(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic OHLCV data for testing.
    Replace with real broker data feed for production.
    """
    np.random.seed(seed)
    dates  = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close  = 4500 + np.cumsum(np.random.randn(n) * 2)
    high   = close + np.abs(np.random.randn(n) * 3)
    low    = close - np.abs(np.random.randn(n) * 3)
    open_  = close + np.random.randn(n) * 1
    volume = np.abs(np.random.randn(n) * 5000 + 10000)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VP-ORB Algorithmic Trading System")
    parser.add_argument("--mode",       default="backtest", choices=["backtest", "paper"])
    parser.add_argument("--symbol",     default="ES")
    parser.add_argument("--capital",    default=2000.0, type=float)
    args = parser.parse_args()

    system = VPORBSystem(instrument=args.symbol, equity=args.capital)

    if args.mode == "backtest":
        log.info("Generating synthetic data for backtest demo...")
        df = generate_synthetic_ohlcv(n=5000)
        stats = system.run_backtest(df)
        print("\n=== PERFORMANCE SUMMARY ===")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k:<25} {v:.4f}")
            else:
                print(f"  {k:<25} {v}")

    elif args.mode == "paper":
        log.info(f"Paper trading mode — {args.symbol}. Connect your data feed.")
        log.info("Call system.on_bar(df) on each new bar from your data provider.")