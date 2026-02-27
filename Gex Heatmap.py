#!/usr/bin/env python3
"""
Net GEX Heat Map — OI-Based Edition
=====================================
Visualizes Net Gamma Exposure (Net GEX) by strike and expiration using
traditional Open Interest (OI) based greek exposure from Unusual Whales.

  Net GEX at strike = call_gamma_oi + put_gamma_oi

  Positive net GEX → dealers long gamma → vol suppression / price pinning
  Negative net GEX → dealers short gamma → vol amplification

Uses only Unusual Whales endpoints defined in api_spec.yaml:
  - GET /api/stock/{ticker}/stock-state
  - GET /api/stock/{ticker}/expiry-breakdown  (optional query: date)
  - GET /api/stock/{ticker}/greek-exposure/strike-expiry  (expiry, optional date)

Author: William Stewart
Date: February 2026
API: Unusual Whales (https://api.unusualwhales.com/docs)
"""

import os
import sys
import time
import logging
import argparse
import tkinter as tk
from tkinter import ttk, font as tkfont, messagebox
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Configure stdout for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import requests
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.style import Style
from rich.text import Text
from dotenv import load_dotenv

# Add Ticker Search folder to path so we can import ticker_search
_TICKER_SEARCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Ticker Search')
if _TICKER_SEARCH_DIR not in sys.path:
    sys.path.insert(0, _TICKER_SEARCH_DIR)
from ticker_search import stock_search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load .env: try script dir, then parent Python Scripts directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_env_paths = [
    os.path.join(_script_dir, '.env'),
    os.path.join(os.path.dirname(_script_dir), '.env'),
]
for _p in _env_paths:
    if os.path.isfile(_p):
        load_dotenv(_p)
        break
UW_TOKEN = (
    os.getenv("UNUSUAL_WHALES_API_TOKEN")
    or os.getenv("UNUSUAL_WHALES_API_KEY")
)
if not UW_TOKEN:
    print("ERROR: Unusual Whales API key not found.")
    print("  Set UNUSUAL_WHALES_API_TOKEN or UNUSUAL_WHALES_API_KEY in your .env file.")
    print("  Checked .env at:", _env_paths)
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL         = "https://api.unusualwhales.com/api"
REQUEST_TIMEOUT  = 30
RATE_LIMIT_SLEEP = 0.5   # Seconds between requests (~120 req/min limit)

DEFAULT_EXPIRIES = 6     # Today (or last market open) + next 5 market dates
DEFAULT_STRIKES  = 20    # Strikes above/below ATM to include


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SpotData:
    ticker: str
    price: float
    timestamp: datetime


@dataclass
class DirectionalGEX:
    """
    Directionalized Net GEX for a single strike.

    Gamma is always positive across the entire chain. The dealer's gamma sign
    depends only on whether they are long or short the contract — not on
    whether it's a call or put.

    Net GEX at strike =
        call_gamma_ask  (negative: dealer short calls, customer bought ask)
      + call_gamma_bid  (positive: dealer long calls, customer sold bid)
      + put_gamma_ask   (negative: dealer short puts, customer bought ask)
      + put_gamma_bid   (positive: dealer long puts, customer sold bid)

    Negative total → customers net BUYING  (dealer short gamma → vol amplification)
    Positive total → customers net SELLING (dealer long gamma  → vol suppression)
    """
    strike: float
    expiry_date: str
    call_gamma_ask: float = 0.0
    call_gamma_bid: float = 0.0
    put_gamma_ask:  float = 0.0
    put_gamma_bid:  float = 0.0
    call_gamma_oi:  float = 0.0
    put_gamma_oi:   float = 0.0

    @property
    def net_gex_directional(self) -> float:
        return self.call_gamma_ask + self.call_gamma_bid + self.put_gamma_ask + self.put_gamma_bid

    @property
    def net_gex_oi(self) -> float:
        return self.call_gamma_oi + self.put_gamma_oi

    @property
    def customer_buying_pressure(self) -> float:
        return self.call_gamma_ask + self.put_gamma_ask

    @property
    def customer_selling_pressure(self) -> float:
        return self.call_gamma_bid + self.put_gamma_bid


# ============================================================================
# MARKET DATE HELPERS
# ============================================================================

def get_last_market_open(dt: Optional[date] = None) -> date:
    """Return the most recent market-open date (weekday). If today is weekend, return previous Friday."""
    d = (dt or datetime.now().date())
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d -= timedelta(days=1)
    return d


# ============================================================================
# UNUSUAL WHALES API CLIENT
# ============================================================================

class UnusualWhalesAPI:
    """
    Unusual Whales API client. Endpoints and params follow api_spec.yaml only.

    Paths (from spec):
      /api/stock/{ticker}/stock-state
      /api/stock/{ticker}/expiry-breakdown
      /api/stock/{ticker}/spot-exposures/expiry-strike
      /api/stock/{ticker}/greek-exposure/strike-expiry
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Accept':        'application/json'
        })
        self._cache: Dict[str, object] = {}

    def _make_request(self, endpoint: str, params: Optional[Dict] = None, retries: int = 2) -> Dict:
        url       = f"{BASE_URL}{endpoint}"
        cache_key = f"{endpoint}|{params}"

        if cache_key in self._cache:
            logger.debug(f"Cache hit: {endpoint}")
            return self._cache[cache_key]

        for attempt in range(retries + 1):
            try:
                logger.debug(f"GET {endpoint} params={params}")
                resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Sleeping {retry_after}s (attempt {attempt + 1})")
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                time.sleep(RATE_LIMIT_SLEEP)

                data = resp.json()
                self._cache[cache_key] = data
                return data

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else None
                if status == 401:
                    logger.error("Authentication failed. Verify your UW API key.")
                    raise
                elif status == 403:
                    logger.error("Forbidden. Endpoint may require Professional/Enterprise tier.")
                    raise
                elif status == 404:
                    logger.error(f"Not found: {endpoint}. Verify endpoint in OpenAPI spec.")
                    raise
                elif status == 422:
                    logger.error(f"Unprocessable Entity: {endpoint}. Check parameter format.")
                    raise
                else:
                    if attempt < retries:
                        logger.warning(f"HTTP {status}, retrying ({attempt + 1}/{retries})...")
                        time.sleep(2 ** attempt)
                        continue
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < retries:
                    logger.warning(f"Request error: {e}, retrying ({attempt + 1}/{retries})...")
                    time.sleep(2 ** attempt)
                    continue
                raise

        raise RuntimeError(f"Max retries exceeded for {endpoint}")

    def get_spot_price(self, ticker: str) -> SpotData:
        """GET /api/stock/{ticker}/stock-state → last OHLCV tick."""
        data  = self._make_request(f"/stock/{ticker}/stock-state")
        inner = data.get('data', data)
        price = float(inner.get('close', inner.get('last', 0)))
        if price == 0:
            raise ValueError(f"Could not parse price from stock-state: {inner}")
        return SpotData(ticker=ticker.upper(), price=price, timestamp=datetime.now())

    def get_expirations(self, ticker: str, limit: int = 10, as_of_date: Optional[date] = None) -> List[str]:
        """GET /api/stock/{ticker}/expiry-breakdown → nearest N future expiry dates with OI.
        as_of_date: use this trading day (e.g. last market open); API returns expirations for that day."""
        params: Dict = {}
        if as_of_date is not None:
            params["date"] = as_of_date.strftime("%Y-%m-%d")
        data  = self._make_request(f"/stock/{ticker}/expiry-breakdown", params=params if params else None)
        logger.debug("expiry-breakdown raw keys: %s", data.keys() if isinstance(data, dict) else type(data))
        rows  = data.get('data', data) if isinstance(data, dict) else data
        if not isinstance(rows, (list, tuple)):
            logger.debug("expiry-breakdown 'rows' is not list: type=%s, value=%s", type(rows).__name__, rows)
            rows = []
        if rows:
            logger.debug("expiry-breakdown first row: %s", rows[0] if rows else None)
        ref_date = as_of_date or get_last_market_open()
        valid = []

        for row in rows:
            if isinstance(row, dict):
                exp_str = str(row.get('expires') or row.get('expiry') or row.get('expiration') or row.get('expiry_date') or '')
                oi      = int(row.get('open_interest', row.get('oi', 0)) or 0)
            else:
                exp_str = str(row)
                oi      = 1
            logger.debug("expiry row: exp_str=%r oi=%s", exp_str, oi)
            if not exp_str or exp_str == '0':
                continue
            try:
                exp_date = datetime.strptime(exp_str[:10], '%Y-%m-%d').date()
                if exp_date >= ref_date:
                    valid.append(exp_str[:10])
                else:
                    logger.debug("skip past expiry %s", exp_str)
            except ValueError:
                try:
                    exp_date = datetime.strptime(exp_str[:10], '%Y-%m-%d').date()
                    if exp_date >= ref_date:
                        valid.append(exp_str[:10])
                except Exception:
                    logger.debug("skip unparseable expiry %r", exp_str)
                    continue

        result = sorted(set(valid))[:limit]
        logger.debug("get_expirations valid list: %s", result)
        return result

    def get_spot_exposures_by_strike_expiry(
        self,
        ticker: str,
        expiry_dates: List[str],
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        GET /api/stock/{ticker}/spot-exposures/expiry-strike
        Per-strike, per-expiry spot GEX with directionalized bid/ask breakdown.

        ask fields → negative (dealer short, customer bought)
        bid fields → positive (dealer long, customer sold)
        """
        # api_spec: expirations[] (required), min_strike, max_strike, limit (max 500), page
        params: Dict = {'expirations[]': expiry_dates, 'limit': 500}
        if min_strike is not None:
            params['min_strike'] = str(min_strike)
        if max_strike is not None:
            params['max_strike'] = str(max_strike)

        all_rows: List[Dict] = []
        page = 1
        while True:
            params_page = dict(params)
            if page > 1:
                params_page['page'] = page
            data = self._make_request(f"/stock/{ticker}/spot-exposures/expiry-strike", params=params_page)
            rows = data.get('data', data) if isinstance(data, dict) else data
            if not isinstance(rows, (list, tuple)):
                rows = []
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < 500:
                break
            page += 1
            time.sleep(RATE_LIMIT_SLEEP)

        if not all_rows:
            logger.warning(f"No spot-exposure data returned for {ticker}")
            return pd.DataFrame()

        logger.debug("spot-exposures first row keys: %s", list(all_rows[0].keys()) if isinstance(all_rows[0], dict) else None)

        records = []
        for row in all_rows:
            exp_raw = row.get('expires') or row.get('expiry') or row.get('expiration') or row.get('expiry_date') or ''
            exp_str = str(exp_raw)[:10] if exp_raw else ''
            strike_val = row.get('price', row.get('strike', 0))
            strike_rounded = round(float(strike_val), 2)
            records.append({
                'strike':         strike_rounded,
                'expiry_date':    exp_str,
                'call_gamma_ask': float(row.get('call_gamma_ask', 0)),
                'call_gamma_bid': float(row.get('call_gamma_bid', 0)),
                'put_gamma_ask':  float(row.get('put_gamma_ask', 0)),
                'put_gamma_bid':  float(row.get('put_gamma_bid', 0)),
                'call_gamma_oi':  float(row.get('call_gamma_oi', 0)),
                'put_gamma_oi':   float(row.get('put_gamma_oi', 0)),
            })

        return pd.DataFrame(records)

    def get_greek_exposure_by_strike_expiry(
        self, ticker: str, expiry_date: str, as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        GET /api/stock/{ticker}/greek-exposure/strike-expiry?expiry=YYYY-MM-DD&date=YYYY-MM-DD
        OI-based greek exposure per strike for a given expiry on a given trading date.
        Passing date ensures we get data for the correct market day.
        """
        params: Dict = {'expiry': expiry_date}
        if as_of_date is not None:
            params['date'] = as_of_date.isoformat()
        data = self._make_request(
            f"/stock/{ticker}/greek-exposure/strike-expiry",
            params=params
        )
        rows = data.get('data', data) if isinstance(data, dict) else data

        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            exp_val = row.get('expiry', expiry_date)
            exp_str = str(exp_val)[:10] if exp_val else expiry_date
            strike_val = row.get('strike', 0)
            strike_rounded = round(float(strike_val), 2)
            # API returns call_gex / put_gex (not call_gamma / put_gamma)
            call_g = row.get('call_gex', row.get('call_gamma', 0))
            put_g  = row.get('put_gex',  row.get('put_gamma',  0))
            records.append({
                'strike':      strike_rounded,
                'expiry_date': exp_str,
                'call_gamma':  float(call_g) if call_g not in (None, '') else 0.0,
                'put_gamma':   float(put_g)  if put_g  not in (None, '') else 0.0,
            })

        return pd.DataFrame(records)


# ============================================================================
# GEX COMPUTATION ENGINE
# ============================================================================

class DirectionalGEXCalculator:
    """
    Computes Net GEX using directionalized volume from the UW spot-exposures
    endpoint, with OI-based fallback.

    Directionalized Net GEX at a strike:
      call_gamma_ask + call_gamma_bid + put_gamma_ask + put_gamma_bid

    Traditional OI-based Net GEX (fallback):
      call_gamma_oi + put_gamma_oi
    """

    @staticmethod
    def compute_from_spot_exposures(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # Aggregate duplicate (strike, expiry_date) rows (API may return multiple timestamps)
        gcols = ['call_gamma_ask', 'call_gamma_bid', 'put_gamma_ask', 'put_gamma_bid',
                 'call_gamma_oi', 'put_gamma_oi']
        agg = {c: 'sum' for c in gcols if c in df.columns}
        if agg:
            df = df.groupby(['strike', 'expiry_date'], as_index=False).agg(agg)

        df['net_gex_dir'] = (
            df['call_gamma_ask'] +
            df['call_gamma_bid'] +
            df['put_gamma_ask']  +
            df['put_gamma_bid']
        )
        df['net_gex_oi']       = df['call_gamma_oi'] + df['put_gamma_oi']
        df['customer_buying']  = df['call_gamma_ask'] + df['put_gamma_ask']
        df['customer_selling'] = df['call_gamma_bid'] + df['put_gamma_bid']

        total_bid = df['call_gamma_bid'].abs() + df['put_gamma_bid'].abs()
        total_ask = df['call_gamma_ask'].abs() + df['put_gamma_ask'].abs()
        df['bid_ask_ratio'] = total_bid / total_ask.replace(0, float('nan'))

        return df

    @staticmethod
    def compute_from_oi_greeks(greek_df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
        if greek_df.empty:
            return greek_df

        greek_df['net_gex_oi']       = greek_df['call_gamma'] + greek_df['put_gamma']
        greek_df['net_gex_dir']      = greek_df['net_gex_oi']
        greek_df['call_gamma_ask']   = 0.0
        greek_df['call_gamma_bid']   = 0.0
        greek_df['put_gamma_ask']    = 0.0
        greek_df['put_gamma_bid']    = 0.0
        greek_df['customer_buying']  = 0.0
        greek_df['customer_selling'] = 0.0
        greek_df['bid_ask_ratio']    = float('nan')

        return greek_df

    @staticmethod
    def filter_strikes_around_spot(df: pd.DataFrame, spot_price: float, n_strikes: int = 5) -> pd.DataFrame:
        if df.empty:
            return df

        strikes    = df['strike'].unique()
        atm_strike = min(strikes, key=lambda s: abs(s - spot_price))
        above      = sorted([s for s in strikes if s > atm_strike])[:n_strikes]
        below      = sorted([s for s in strikes if s < atm_strike], reverse=True)[:n_strikes]
        keep       = set(above + below + [atm_strike])

        return df[df['strike'].isin(keep)].sort_values('strike', ascending=False).copy()


# ============================================================================
# RENDERING ENGINE
# ============================================================================

class GEXHeatMapRenderer:
    """Console renderer using rich for a color-coded Net GEX heat map."""

    def __init__(self):
        self.console = Console()

    def _format_gex(self, value: float) -> Tuple[str, Style]:
        abs_val = abs(value)

        if abs_val >= 1e9:
            display = f"{value / 1e9:+.2f} B"
        elif abs_val >= 1e6:
            display = f"{value / 1e6:+.2f} M"
        elif abs_val >= 1e3:
            display = f"{value / 1e3:+.2f} K"
        elif abs_val > 0:
            display = f"{value:+.0f}"
        else:
            display = "—"

        if value > 0:
            if abs_val > 1e9:
                style = Style(color="bright_magenta", bold=True)
            elif abs_val > 500e6:
                style = Style(color="bright_green", bold=True)
            elif abs_val > 100e6:
                style = Style(color="green")
            elif abs_val > 10e6:
                style = Style(color="green", dim=False)
            else:
                style = Style(color="green", dim=True)
        elif value < 0:
            if abs_val > 1e9:
                style = Style(color="bright_red", bold=True, underline=True)
            elif abs_val > 500e6:
                style = Style(color="bright_red", bold=True)
            elif abs_val > 100e6:
                style = Style(color="red")
            elif abs_val > 10e6:
                style = Style(color="red", dim=False)
            else:
                style = Style(color="red", dim=True)
        else:
            style = Style(color="bright_black")

        return display, style

    def _format_ratio(self, ratio: float) -> Tuple[str, Style]:
        if pd.isna(ratio) or ratio == 0:
            return "N/A", Style(color="bright_black")
        display = f"{ratio:.2f}"
        if ratio > 1.5:
            return display, Style(color="bright_green")
        elif ratio < 0.67:
            return display, Style(color="bright_red")
        return display, Style(color="yellow")

    def render(
        self,
        ticker: str,
        spot_price: float,
        gex_matrix: pd.DataFrame,
        expiry_dates: List[str],
        use_directional: bool = True,
        show_decomposition: bool = True,
    ):
        gex_col = 'net_gex_dir' if use_directional else 'net_gex_oi'
        label   = "Directionalized Volume" if use_directional else "Open Interest"

        # Wrap the full heat map output in a pager so you can scroll
        # through many strikes cleanly in the terminal.
        with self.console.pager(styles=True):
            self.console.print()
            self.console.rule(f"[bold cyan]Net GEX Heat Map — {ticker}[/bold cyan]", style="cyan")
            self.console.print(
                f"  [bold]Method:[/bold] {label}  │  "
                f"[bold]Underlying:[/bold] [yellow]${spot_price:.2f}[/yellow]  │  "
                f"[bold]Time:[/bold] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self.console.print()

            if gex_col not in gex_matrix.columns:
                logger.error(f"Column '{gex_col}' not found. Available: {list(gex_matrix.columns)}")
                return

            pivot = gex_matrix.pivot_table(
                index='strike', columns='expiry_date', values=gex_col, fill_value=0
            )
            valid_expiries = [e for e in expiry_dates if e in pivot.columns]
            pivot = pivot[valid_expiries].sort_index(ascending=False)

            atm_strike = min(pivot.index, key=lambda s: abs(s - spot_price))

            # ── Main GEX heat map table ─────────────────────────────────────
            table = Table(
                title=f"[bold]Net GEX by Strike × Expiry ({label})[/bold]",
                show_header=True,
                header_style="bold magenta",
                border_style="dim",
                pad_edge=True,
            )
            table.add_column("Strike", style="cyan", justify="right", width=10)
            for exp in valid_expiries:
                exp_dt    = datetime.strptime(exp, '%Y-%m-%d')
                col_label = exp_dt.strftime('%b %d')
                table.add_column(col_label, justify="right", width=14)
            table.add_column("Row Total", justify="right", width=14, style="bold")

            for strike in pivot.index:
                # Show whole dollars when strikes are integral; otherwise show 2 decimals.
                if abs(strike - round(strike)) < 1e-6:
                    strike_label = f"${int(round(strike))}"
                else:
                    strike_label = f"${strike:.2f}"
                strike_text = (
                    Text(f"★{strike_label}", style="bold yellow")
                    if strike == atm_strike
                    else Text(strike_label)
                )
                cells     = [strike_text]
                row_total = 0.0

                for exp in valid_expiries:
                    val = pivot.loc[strike, exp]
                    row_total += val
                    display, style = self._format_gex(val)
                    cells.append(Text(display, style=style))

                total_display, total_style = self._format_gex(row_total)
                cells.append(Text(total_display, style=total_style))
                table.add_row(*cells)

            self.console.print(table)

            # ── Bid/Ask decomposition table ─────────────────────────────────
            if show_decomposition and use_directional:
                self.console.print()
                decomp = Table(
                    title="[bold]Bid/Ask Side Decomposition (All Expiries Combined)[/bold]",
                    show_header=True,
                    header_style="bold blue",
                    border_style="dim",
                )
                decomp.add_column("Strike",               style="cyan",  justify="right", width=10)
                decomp.add_column("Ask-Side (Cust Buy)",                 justify="right", width=16)
                decomp.add_column("Bid-Side (Cust Sell)",                justify="right", width=16)
                decomp.add_column("Net Dir. GEX",                        justify="right", width=14)
                decomp.add_column("Bid/Ask Ratio",                       justify="right", width=13)

                agg = gex_matrix.groupby('strike').agg({
                    'customer_buying':  'sum',
                    'customer_selling': 'sum',
                    'net_gex_dir':      'sum',
                    'bid_ask_ratio':    'mean',
                }).sort_index(ascending=False)

                for strike in agg.index:
                    if strike not in pivot.index:
                        continue
                    row_data     = agg.loc[strike]
                    strike_label = f"★${strike:.0f}" if strike == atm_strike else f"${strike:.0f}"
                    strike_style = "bold yellow"       if strike == atm_strike else "cyan"

                    buy_val,   buy_style   = self._format_gex(row_data['customer_buying'])
                    sell_val,  sell_style  = self._format_gex(row_data['customer_selling'])
                    net_val,   net_style   = self._format_gex(row_data['net_gex_dir'])
                    ratio_val, ratio_style = self._format_ratio(row_data['bid_ask_ratio'])

                    decomp.add_row(
                        Text(strike_label, style=strike_style),
                        Text(buy_val,      style=buy_style),
                        Text(sell_val,     style=sell_style),
                        Text(net_val,      style=net_style),
                        Text(ratio_val,    style=ratio_style),
                    )

                self.console.print(decomp)

            # ── Summary stats ───────────────────────────────────────────────
            self.console.print()
            total_dir     = gex_matrix[gex_col].sum()
            total_buying  = gex_matrix['customer_buying'].sum()  if 'customer_buying'  in gex_matrix.columns else 0
            total_selling = gex_matrix['customer_selling'].sum() if 'customer_selling' in gex_matrix.columns else 0

            total_fmt, total_sty = self._format_gex(total_dir)
            buy_fmt,   _         = self._format_gex(total_buying)
            sell_fmt,  _         = self._format_gex(total_selling)

            self.console.print(f"  [bold]Total Net GEX ({label}):[/bold] ", end="")
            self.console.print(Text(total_fmt, style=total_sty))

            if use_directional:
                self.console.print(f"  [bold]Customer Buying  (ask-side):[/bold] [red]{buy_fmt}[/red]")
                self.console.print(f"  [bold]Customer Selling (bid-side):[/bold] [green]{sell_fmt}[/green]")

            # ── Key levels ─────────────────────────────────────────────────
            agg_strikes = gex_matrix.groupby('strike')[gex_col].sum()
            if not agg_strikes.empty:
                max_pos_strike = agg_strikes.idxmax()
                max_neg_strike = agg_strikes.idxmin()
                max_pos_val, _ = self._format_gex(agg_strikes[max_pos_strike])
                max_neg_val, _ = self._format_gex(agg_strikes[max_neg_strike])

                self.console.print()
                self.console.print(f"  [bold]Largest Positive GEX:[/bold] [green]${max_pos_strike:.0f}[/green] ({max_pos_val})")
                self.console.print(f"  [bold]Largest Negative GEX:[/bold] [red]${max_neg_strike:.0f}[/red] ({max_neg_val})")

            # ── Market regime interpretation ───────────────────────────────
            self.console.print()
            if total_dir > 0:
                self.console.print("[bold green]→ Net POSITIVE gamma (dealer long gamma)[/bold green]")
                self.console.print("  Customers are net SELLING options → dealers long gamma")
                self.console.print("  Dealers hedge by: buying dips, selling rallies → vol suppression / pinning")
            else:
                self.console.print("[bold red]→ Net NEGATIVE gamma (dealer short gamma)[/bold red]")
                self.console.print("  Customers are net BUYING options → dealers short gamma")
                self.console.print("  Dealers hedge by: selling dips, buying rallies → vol amplification")

            if use_directional and total_buying != 0:
                sell_buy_pct = abs(total_selling / total_buying) * 100
                self.console.print(f"  Bid/Ask balance: customer sells = {sell_buy_pct:.0f}% of customer buys")

            self.console.print()
            self.console.print(
                "[dim]★ = ATM strike  │  "
                "Ask-side = customer BUY (dealer SHORT gamma)  │  "
                "Bid-side = customer SELL (dealer LONG gamma)[/dim]"
            )
            self.console.print(
                "[dim]Directionalized volume uses NBBO trade-side attribution, "
                "not naive OI assumptions.[/dim]"
            )
            self.console.print()


# ============================================================================
# GUI HEAT MAP
# ============================================================================

def _format_gex_cell(value: float) -> str:
    """Format GEX value for display (e.g. 15.19 M, -485.65 K)."""
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"{value / 1e9:+.2f} B"
    if abs_val >= 1e6:
        return f"{value / 1e6:+.2f} M"
    if abs_val >= 1e3:
        return f"{value / 1e3:+.2f} K"
    if abs_val > 0:
        return f"{value:+.0f}"
    return "—"


def _gex_cell_bg(value: float, max_abs: float) -> str:
    """Return hex background color: green for positive, red for negative (dark theme)."""
    if max_abs <= 0:
        return "#2d2d2d"
    intensity = min(abs(value) / max_abs, 1.0)
    if value > 0:
        # Green: #1a2e1a (low) -> #0d6b0d (high)
        r = int(26 + (13 - 26) * intensity)
        g = int(46 + (107 - 46) * intensity)
        b = int(26 + (13 - 26) * intensity)
        return f"#{r:02x}{g:02x}{b:02x}"
    elif value < 0:
        # Red: #2e1a1a (low) -> #a52a2a (high)
        r = int(46 + (165 - 46) * intensity)
        g = int(26 + (42 - 26) * intensity)
        b = int(26 + (42 - 26) * intensity)
        return f"#{r:02x}{g:02x}{b:02x}"
    return "#2d2d2d"


class GEXHeatMapGUI:
    """Desktop GUI for Net GEX Heat Map — ticker button, metrics, scrollable color grid."""

    BG_DARK   = "#1e1e1e"
    FG        = "#e0e0e0"
    HEADER_BG = "#252526"
    TICKER_BG = "#0e639c"
    TICKER_FG = "#ffffff"

    def __init__(self, n_expiries: int = 6, n_strikes: int = 20,
                 initial_ticker: Optional[str] = None):
        self.n_expiries = n_expiries  # today + next 5 market dates = 6 columns
        self.n_strikes  = n_strikes
        self.ticker     = (initial_ticker or "SPY").strip().upper()
        self.spot_price = 0.0
        self.gex_matrix = pd.DataFrame()
        self.expiry_dates: List[str] = []
        self.as_of_date: Optional[date] = None  # last market open, for header

        self.root = tk.Tk()
        self.root.title("Net GEX Heat Map")
        self.root.configure(bg=self.BG_DARK)
        self.root.minsize(900, 600)
        self.root.geometry("1100x700")

        self._build_ui()
        self._load_data()

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg=self.HEADER_BG, height=48)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)

        self.ticker_btn = tk.Button(
            top, text=self.ticker, font=("Segoe UI", 11, "bold"),
            bg=self.TICKER_BG, fg=self.TICKER_FG, activebackground="#1177bb",
            relief=tk.FLAT, padx=16, pady=6, cursor="hand2",
            command=self._on_change_ticker,
        )
        self.ticker_btn.pack(side=tk.LEFT, padx=12, pady=8)

        self._date_label = tk.Label(
            top, text="—", fg=self.FG, bg=self.HEADER_BG, font=("Segoe UI", 10),
        )
        self._date_label.pack(side=tk.RIGHT, padx=16, pady=12)
        tk.Button(
            top, text="Net GEX", fg=self.FG, bg=self.HEADER_BG, relief=tk.FLAT,
            font=("Segoe UI", 10), state=tk.DISABLED,
        ).pack(side=tk.RIGHT, padx=4, pady=8)
        refresh_btn = tk.Button(
            top, text="↻ Refresh", fg=self.FG, bg=self.HEADER_BG, relief=tk.FLAT,
            font=("Segoe UI", 10), cursor="hand2", command=self._load_data,
        )
        refresh_btn.pack(side=tk.RIGHT, padx=4, pady=8)

        # Content
        content = tk.Frame(self.root, bg=self.BG_DARK, padx=16, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(
            content, text=f"Net GEX Heat Map - {self.ticker}",
            fg=self.FG, bg=self.BG_DARK, font=("Segoe UI", 16, "bold"),
        )
        title.pack(anchor=tk.W)
        self._title_label = title

        metrics = tk.Frame(content, bg=self.BG_DARK)
        metrics.pack(anchor=tk.W, pady=(8, 12))
        self._mvc_label = tk.Label(
            metrics, text="MVC (—)", fg="#c586c0", bg=self.BG_DARK, font=("Segoe UI", 10),
        )
        self._mvc_label.pack(side=tk.LEFT)
        self._underlying_label = tk.Label(
            metrics, text="Underlying (—)", fg="#569cd6", bg=self.BG_DARK, font=("Segoe UI", 10),
        )
        self._underlying_label.pack(side=tk.LEFT, padx=(20, 0))

        # Scrollable grid
        grid_frame = tk.Frame(content, bg=self.BG_DARK)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(grid_frame, bg=self.BG_DARK, highlightthickness=0)
        vbar = ttk.Scrollbar(grid_frame, orient=tk.VERTICAL, command=canvas.yview)
        hbar = ttk.Scrollbar(grid_frame, orient=tk.HORIZONTAL, command=canvas.xview)

        self._table_frame = tk.Frame(canvas, bg=self.BG_DARK)
        self._table_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._table_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_shift_mousewheel(event):
            canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        self.root.bind("<MouseWheel>", _on_mousewheel)
        self.root.bind("<Shift-MouseWheel>", _on_shift_mousewheel)

    def _on_change_ticker(self):
        new_ticker = stock_search()
        if new_ticker and str(new_ticker).strip():
            self.ticker = str(new_ticker).strip().upper()
            self.ticker_btn.configure(text=self.ticker)
            self._load_data()

    def _load_data(self):
        self._title_label.configure(text=f"Net GEX Heat Map - {self.ticker}")
        self.root.config(cursor="wait")
        self.root.update()
        try:
            spot_price, gex_matrix, expiry_dates, data_date = fetch_gex_data(
                self.ticker, self.n_expiries, self.n_strikes,
            )
            self.spot_price   = spot_price
            self.gex_matrix   = gex_matrix
            self.expiry_dates = expiry_dates
            self.as_of_date   = data_date
            self._date_label.configure(text=data_date.strftime("%b %d, %Y"))
            self._render_grid()
            self._update_metrics()
        except Exception as e:
            messagebox.showerror("GEX Error", str(e))
            logger.exception("GUI load failed")
        finally:
            self.root.config(cursor="")

    def _update_metrics(self):
        self._underlying_label.configure(text=f"Underlying (${self.spot_price:.2f})")
        if 'net_gex_oi' not in self.gex_matrix.columns or self.gex_matrix.empty:
            self._mvc_label.configure(text="MVC (—)")
            return
        idx    = self.gex_matrix['net_gex_oi'].idxmax()
        row    = self.gex_matrix.loc[idx]
        strike = row['strike']
        exp    = row['expiry_date']
        try:
            exp_fmt = datetime.strptime(exp, '%Y-%m-%d').strftime('%b %d, %Y')
        except Exception:
            exp_fmt = exp
        self._mvc_label.configure(text=f"MVC (${strike:.0f} {exp_fmt})")

    def _render_grid(self):
        for w in self._table_frame.winfo_children():
            w.destroy()

        gex_col = 'net_gex_oi'
        if gex_col not in self.gex_matrix.columns or self.gex_matrix.empty:
            tk.Label(
                self._table_frame, text="No data", fg=self.FG, bg=self.BG_DARK,
                font=("Segoe UI", 12),
            ).pack(pady=20)
            return

        pivot = self.gex_matrix.pivot_table(
            index='strike', columns='expiry_date', values=gex_col, fill_value=0, aggfunc='sum'
        )
        valid_expiries = [e for e in self.expiry_dates if e in pivot.columns]
        pivot = pivot[valid_expiries].sort_index(ascending=False)
        # Ensure unique strike index (collapse any float-precision duplicates)
        if pivot.index.duplicated().any():
            pivot = pivot.groupby(level=0).sum()
        max_abs = pivot.abs().values.max() if pivot.size else 1.0

        cell_font = tkfont.Font(family="Segoe UI", size=9)
        header_font = tkfont.Font(family="Segoe UI", size=9, weight="bold")

        # Header row: Strike Price | exp1 | exp2 | ...
        tk.Label(
            self._table_frame, text="Strike Price", fg=self.FG, bg=self.HEADER_BG,
            font=header_font, width=12, anchor=tk.W, padx=6, pady=4,
        ).grid(row=0, column=0, sticky=tk.NSEW, padx=1, pady=1)
        for c, exp in enumerate(valid_expiries, start=1):
            try:
                dt = datetime.strptime(exp, '%Y-%m-%d')
                label = dt.strftime("%b %d, %Y")
            except Exception:
                label = exp
            tk.Label(
                self._table_frame, text=label, fg=self.FG, bg=self.HEADER_BG,
                font=header_font, width=14, anchor=tk.CENTER, padx=6, pady=4,
            ).grid(row=0, column=c, sticky=tk.NSEW, padx=1, pady=1)

        # Find the cell with the largest absolute value per column → one purple cell per expiry
        purple_cells: set = set()
        for exp in valid_expiries:
            col = pivot[exp]
            if col.abs().max() > 0:
                top_strike = col.abs().idxmax()
                purple_cells.add((top_strike, exp))

        atm_strike = min(pivot.index, key=lambda s: abs(s - self.spot_price))
        for r, strike in enumerate(pivot.index, start=1):
            is_atm = strike == atm_strike
            if abs(strike - round(strike)) < 1e-6:
                strike_label = f"${int(round(strike))}"
            else:
                strike_label = f"${strike:.2f}"
            tk.Label(
                self._table_frame, text=strike_label,
                fg="#569cd6" if is_atm else self.FG, bg=self.HEADER_BG if is_atm else self.BG_DARK,
                font=header_font if is_atm else cell_font,
                width=12, anchor=tk.E, padx=6, pady=3,
            ).grid(row=r, column=0, sticky=tk.NSEW, padx=1, pady=1)
            for c, exp in enumerate(valid_expiries, start=1):
                val = pivot.loc[strike, exp]
                txt = _format_gex_cell(val)
                is_purple = (strike, exp) in purple_cells
                bg = "#6a0dad" if is_purple else _gex_cell_bg(val, max_abs)
                fg = "#ffffff"
                tk.Label(
                    self._table_frame, text=txt, fg=fg, bg=bg,
                    font=cell_font, width=14, anchor=tk.CENTER, padx=6, pady=3,
                ).grid(row=r, column=c, sticky=tk.NSEW, padx=1, pady=1)

        for i in range(len(valid_expiries) + 1):
            self._table_frame.columnconfigure(i, weight=0 if i == 0 else 1)
        for i in range(len(pivot.index) + 1):
            self._table_frame.rowconfigure(i, weight=0)

    def run(self):
        self.root.mainloop()


# ============================================================================
# DATA FETCH (for GUI or CLI)
# ============================================================================

def fetch_gex_data(
    ticker: str,
    n_expiries: int,
    n_strikes: int,
    as_of_date: Optional[date] = None,
) -> Tuple[float, pd.DataFrame, List[str], date]:
    """
    Fetch OI-based Net GEX data for a ticker.
    Returns (spot_price, gex_matrix, expiry_dates, data_date).

    Uses GET /stock/{ticker}/greek-exposure/strike-expiry for each expiry —
    one call per expiry date, giving full-chain call_gamma + put_gamma per strike.
    Net GEX = call_gamma_oi + put_gamma_oi.
    """
    api       = UnusualWhalesAPI(UW_TOKEN)
    calc      = DirectionalGEXCalculator()
    data_date = as_of_date or get_last_market_open()

    logger.info(f"Fetching spot price for {ticker}...")
    spot = api.get_spot_price(ticker)
    logger.info(f"  Underlying: ${spot.price:.2f}")

    logger.info("Fetching expiration dates (as of %s)...", data_date.isoformat())
    expiry_dates = api.get_expirations(ticker, limit=n_expiries, as_of_date=data_date)
    if not expiry_dates:
        raise ValueError(f"No valid expirations found for {ticker}")
    expiry_dates = expiry_dates[:n_expiries]
    logger.info(f"  Expiries: {', '.join(expiry_dates)}")

    logger.info("Fetching OI-based greek exposure (%d expiries, date=%s)...",
                len(expiry_dates), data_date.isoformat())
    frames = []
    for exp in expiry_dates:
        greek_df = api.get_greek_exposure_by_strike_expiry(ticker, exp, as_of_date=data_date)
        if not greek_df.empty:
            frames.append(greek_df)
            logger.info("  %s → %d strikes", exp, greek_df['strike'].nunique())
        else:
            logger.warning("  %s → no data returned", exp)
    if not frames:
        raise ValueError("No GEX data returned. Check API subscription tier.")

    all_gex = pd.concat(frames, ignore_index=True)
    all_gex = calc.compute_from_oi_greeks(all_gex, spot.price)

    # Normalize strike precision and collapse duplicates
    all_gex["strike"] = all_gex["strike"].round(2)
    gcols = [c for c in all_gex.columns if c not in ("strike", "expiry_date")]
    if gcols:
        all_gex = all_gex.groupby(["strike", "expiry_date"], as_index=False)[gcols].sum()

    if all_gex.empty:
        raise ValueError("No strikes after processing.")

    # Keep only n_strikes above and below ATM
    filtered = calc.filter_strikes_around_spot(all_gex, spot.price, n_strikes)
    if filtered.empty:
        raise ValueError("No strikes in range after ATM filter.")

    return (spot.price, filtered, expiry_dates, data_date)


def run_gex_analysis(ticker: str, n_expiries: int, n_strikes: int) -> None:
    """Run the full OI-based GEX pipeline for a given ticker (terminal output)."""
    try:
        spot_price, gex_matrix, expiry_dates, _ = fetch_gex_data(
            ticker, n_expiries, n_strikes
        )
    except ValueError as e:
        logger.error(str(e))
        return
    renderer = GEXHeatMapRenderer()
    renderer.render(
        ticker=ticker,
        spot_price=spot_price,
        gex_matrix=gex_matrix,
        expiry_dates=expiry_dates,
        use_directional=False,
        show_decomposition=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Net GEX Heat Map — OI-Based (Unusual Whales API)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
By default opens the desktop GUI (ticker search then heat map). Use --no-gui for terminal output.

  --ticker SYMBOL   Ticker to analyze (e.g. SPY)
  --expiries N      Number of expirations — today + next N-1 (default: 6)
  --strikes N       Strikes above/below ATM (default: 20)
  --no-gui          Force terminal output
  --debug           Enable verbose debug logging
        """
    )
    parser.add_argument('--ticker',   type=str, default=None,          help='Ticker symbol (e.g. SPY).')
    parser.add_argument('--expiries', type=int, default=DEFAULT_EXPIRIES, help='Number of expirations (default: 6)')
    parser.add_argument('--strikes',  type=int, default=DEFAULT_STRIKES,  help='Strikes above/below ATM (default: 20)')
    parser.add_argument('--no-gui',   action='store_true',             help='Force terminal output')
    parser.add_argument('--debug',    action='store_true',             help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    use_gui = not args.no_gui

    if args.ticker:
        ticker = args.ticker.strip().upper()
    else:
        ticker = stock_search()
        if not ticker:
            print("No ticker selected. Exiting.")
            sys.exit(0)
        ticker = ticker.strip().upper()

    if use_gui:
        app = GEXHeatMapGUI(
            n_expiries=args.expiries,
            n_strikes=args.strikes,
            initial_ticker=ticker,
        )
        app.run()
        return

    # Terminal output
    print("Net GEX Heat Map — OI-Based Edition")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nRunning GEX analysis for: {ticker}")
    print("=" * 60)
    try:
        run_gex_analysis(ticker=ticker, n_expiries=args.expiries, n_strikes=args.strikes)
    except KeyboardInterrupt:
        logger.info("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == '__main__':
    main()
