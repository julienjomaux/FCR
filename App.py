
# app.py
import io
import requests
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from calendar import month_abbr
from datetime import date

st.set_page_config(page_title="FCR Capacity Price Heatmap", layout="wide")

YEARS = [2021, 2022, 2023, 2024, 2025]
BASE_YEAR_URL = (
    "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/"
    "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_{y}-01-01_{y}-12-31.xlsx"
)
BASE_MONTH_URL = (
    "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/"
    "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_{y}-{m:02d}-01_{y}-{m:02d}-31.xlsx"
)

COUNTRY_RENAME = {
    'BE': 'BELGIUM', 'BELGIUM': 'BELGIUM',
    'DE': 'GERMANY', 'GERMANY': 'GERMANY',
    'FR': 'FRANCE', 'FRANCE': 'FRANCE',
    'NL': 'NETHERLANDS', 'NETHERLANDS': 'NETHERLANDS',
    'AT': 'AUSTRIA', 'AUSTRIA': 'AUSTRIA',
    'SI': 'SLOVENIA', 'SLOVENIA': 'SLOVENIA',
    'DK': 'DENMARK', 'DENMARK': 'DENMARK',
    'CH': 'SWITZERLAND', 'SWITZERLAND': 'SWITZERLAND',
    # extend as needed
}

def harmonize_country(name: str) -> str:
    name = str(name).upper().strip()
    return COUNTRY_RENAME.get(name, name)

def product_bin_label(product: str) -> str:
    try:
        val = int(str(product).strip())
        return f"{val} to {val+4}"
    except Exception:
        return str(product)

def _get(url: str, timeout=60):
    """Try GET with verify=True then fallback to verify=False. Return bytes or None."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and len(r.content) > 10000:
            return r.content
        # fallback (some servers have TLS chain quirks)
        r = requests.get(url, timeout=timeout, verify=False)
        if r.status_code == 200 and len(r.content) > 10000:
            return r.content
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def download_year_df(year: int) -> pd.DataFrame | None:
    """
    Try full-year file first. If not available, fall back to concatenating monthly files.
    Returns a cleaned DataFrame or None.
    """
    # 1) Try full-year
    year_url = BASE_YEAR_URL.format(y=year)
    blob = _get(year_url)
    if blob:
        try:
            df = pd.read_excel(io.BytesIO(blob), engine="openpyxl")
            if not df.empty and 'DATE_FROM' in df.columns:
                return _clean_dates(df)
        except Exception:
            pass

    # 2) Fall back to monthly files
    # For the current year, only fetch months up to today (to avoid empty future months).
    last_month = 12
    today = date.today()
    if year == today.year:
        last_month = today.month

    monthly_frames = []
    for m in range(1, last_month + 1):
        m_url = BASE_MONTH_URL.format(y=year, m=m)
        blob = _get(m_url)
        if not blob:
            # Some months might use 30 days (e.g., 2025-04-01_2025-04-30). Try day=30 fallback.
            m_url_30 = m_url.replace("-31.xlsx", "-30.xlsx")
            blob = _get(m_url_30)
        if not blob:
            continue
        try:
            dfm = pd.read_excel(io.BytesIO(blob), engine="openpyxl")
            if dfm is not None and not dfm.empty and 'DATE_FROM' in dfm.columns:
                monthly_frames.append(dfm)
        except Exception:
            continue

    if not monthly_frames:
        return None

    df_all = pd.concat(monthly_frames, ignore_index=True)
    return _clean_dates(df_all)

def _clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['DATE'] = pd.to_datetime(df['DATE_FROM'], dayfirst=True, errors='coerce')
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month
    df['MONTH_NAME'] = df['DATE'].dt.strftime('%b')
    return df

def extract_countries_from_df(df: pd.DataFrame) -> list[str]:
    candidates = set()
    for col in df.columns:
        if str(col).endswith('SETTLEMENTCAPACITY_PRICE_[EUR/MW]'):
            code = str(col).split('_')[0]
            candidates.add(harmonize_country(code))
    return sorted(candidates)

def find_price_column_for_country(df: pd.DataFrame, country: str) -> str | None:
    keys = []
    for col in df.columns:
        col_str = str(col)
        if col_str.endswith('SETTLEMENTCAPACITY_PRICE_[EUR/MW]'):
            prefix = col_str.split('_')[0]
            if harmonize_country(prefix) == country:
                keys.append(col_str)
    return keys[0] if keys else None

def build_heatmap_for(df: pd.DataFrame, year: int, country: str):
    year_df = df[df['YEAR'] == year].copy()
    if year_df.empty:
        return None, None, None

    price_col = find_price_column_for_country(year_df, country)
    if not price_col:
        return None, None, None

    year_df[price_col] = pd.to_numeric(year_df[price_col], errors='coerce')
    year_df['PRODUCTNAME'] = year_df['PRODUCTNAME'].astype(str)

    products = sorted(
        year_df['PRODUCTNAME'].dropna().unique().tolist(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))
    )
    if not products:
        return None, None, None

    g = (year_df
         .dropna(subset=[price_col])
         .groupby(['MONTH_NAME', 'PRODUCTNAME'])[price_col]
         .mean()
         .reset_index())

    months_label = [month_abbr[m] for m in range(1, 12 + 1)]
    all_months = pd.DataFrame({'MONTH_NAME': months_label})
    all_products = pd.DataFrame({'PRODUCTNAME': products})
    all_months['__k'], all_products['__k'] = 1, 1
    full_index = pd.merge(all_months, all_products, on='__k').drop(columns='__k')

    merged = pd.merge(full_index, g, on=['MONTH_NAME', 'PRODUCTNAME'], how='left')
    heatmap_data = merged.pivot(index='MONTH_NAME', columns='PRODUCTNAME', values=price_col)
    heatmap_data = heatmap_data.reindex(index=months_label, columns=products)

    x_labels_bins = [product_bin_label(p) for p in products]
    return heatmap_data, x_labels_bins, months_label

# ---------------- UI ----------------
st.title("FCR Capacity Price Heatmap (€/MW per block)")
st.caption("Source: regelleistung.net – Capacity Market FCR result files")

left, right = st.columns([1, 3])

