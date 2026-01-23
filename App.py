
import os
import glob
import time
from typing import Optional, List, Tuple, Dict

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from calendar import month_abbr

# ---------------- Page setup ----------------
st.set_page_config(page_title="FCR Heatmap — Price, Demand, Import/Export", layout="wide")

# Years you want to expose in the UI (adapt if you add more files)
YEARS = [2021, 2022, 2023, 2024, 2025]

# Where to look for the Excel files.
# The app will first try the current folder, then a ./data subfolder.
FILENAME_PATTERN = "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_{y}.xlsx"
SEARCH_LOCATIONS = ["", "data"]  # "" = current folder

# Country harmonization (extend as needed)
COUNTRY_RENAME = {
    'BE': 'BELGIUM', 'BELGIUM': 'BELGIUM',
    'DE': 'GERMANY', 'GERMANY': 'GERMANY',
    'FR': 'FRANCE', 'FRANCE': 'FRANCE',
    'NL': 'NETHERLANDS', 'NETHERLANDS': 'NETHERLANDS',
    'AT': 'AUSTRIA', 'AUSTRIA': 'AUSTRIA',
    'SI': 'SLOVENIA', 'SLOVENIA': 'SLOVENIA',
    'DK': 'DENMARK', 'DENMARK': 'DENMARK',
    'CH': 'SWITZERLAND', 'SWITZERLAND': 'SWITZERLAND',
}

def harmonize_country(name: str) -> str:
    name = str(name).upper().strip()
    return COUNTRY_RENAME.get(name, name)

def product_bin_label(product: str) -> str:
    """Turns '0' -> '0 to 4', '5' -> '5 to 9', else returns as-is."""
    try:
        val = int(str(product).strip())
        return f"{val} to {val+4}"
    except Exception:
        return str(product)

def find_local_file_for_year(year: int) -> Optional[str]:
    """
    Look for RESULT_OVERVIEW_CAPACITY_MARKET_FCR_{year}.xlsx
    first in the app folder, then in ./data.
    Returns absolute path or None.
    """
    candidates = []
    for loc in SEARCH_LOCATIONS:
        pattern = os.path.join(loc, FILENAME_PATTERN.format(y=year))
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    # If multiple matches, take the newest
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return os.path.abspath(candidates[0])

@st.cache_data(show_spinner=False)
def load_year_df(path: str, mtime: float) -> Optional[pd.DataFrame]:
    """
    Load the given Excel file and return a cleaned DataFrame.
    Cache is keyed by (path, mtime) via arguments.
    """
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception:
        return None

    if df is None or df.empty or 'DATE_FROM' not in df.columns:
        return None

    df = df.copy()
    df['DATE'] = pd.to_datetime(df['DATE_FROM'], dayfirst=True, errors='coerce')
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month
    df['MONTH_NAME'] = df['DATE'].dt.strftime('%b')
    return df

# ---------------- Metric specifications ----------------
METRICS: Dict[str, Dict[str, Optional[str]]] = {
    "PRICE": {
        "label": "Settlement Capacity Price",
        "suffix": "SETTLEMENTCAPACITY_PRICE_[EUR/MW]",
        "cbar_label": "€/MW",
        "cmap": "YlOrRd",
        "center": None,
        "title_suffix": "Average Capacity Price FCR",
    },
    "DEMAND": {
        "label": "Demand",
        "suffix": "DEMAND_[MW]",
        "cbar_label": "MW",
        "cmap": "YlGnBu",
        "center": None,
        "title_suffix": "Average Demand FCR",
    },
    "IMPORT_EXPORT": {
        "label": "Import (−) / Export (+)",
        "suffix": "IMPORT(-)_EXPORT(+)_[MW]",
        "cbar_label": "MW",
        "cmap": "coolwarm",
        "center": 0.0,  # Diverging map centered at 0 to show import(-) vs export(+)
        "title_suffix": "Average Import(−)/Export(+) FCR",
    },
}

def extract_countries_from_df(df: pd.DataFrame) -> List[str]:
    """
    Detect available countries based on any of the metric suffixes.
    We consider columns like 'AT_DEMAND_[MW]' or 'BE_*SETTLEMENTCAPACITY_PRICE_[EUR/MW]'.
    """
    suffixes = [spec["suffix"] for spec in METRICS.values()]
    candidates = set()
    for col in df.columns:
        col_str = str(col)
        for suf in suffixes:
            if col_str.endswith(suf):
                prefix = col_str.split('_')[0]
                candidates.add(harmonize_country(prefix))
                break
    return sorted(candidates)

def find_metric_column_for_country(df: pd.DataFrame, country: str, metric_key: str) -> Optional[str]:
    """
    From df, find the column name matching the selected country and metric.
    Country is tested via the prefix before the first underscore.
    """
    suffix = METRICS[metric_key]["suffix"]
    matches = []
    for col in df.columns:
        col_str = str(col)
        if col_str.endswith(suffix):
            prefix = col_str.split('_')[0]
            if harmonize_country(prefix) == country:
                matches.append(col_str)
    return matches[0] if matches else None

def ensure_product_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure PRODUCTNAME exists. If it's missing, create a single bucket 'ALL'.
    """
    df = df.copy()
    if 'PRODUCTNAME' not in df.columns:
        df['PRODUCTNAME'] = 'ALL'
    return df

def build_heatmap_for(df: pd.DataFrame, year: int, country: str, metric_key: str):
    """
    Returns (heatmap_data, x_labels_bins, months_label, cbar_label, cmap, center, title_suffix)
    - heatmap_data: index: months (Jan..Dec), columns: PRODUCTNAME (sorted)
    - x_labels_bins: pretty x labels (e.g., '0 to 4' for 0/5/... if product names are numeric)
    - months_label: ['Jan', ..., 'Dec']
    """
    year_df = df[df['YEAR'] == year].copy()
    if year_df.empty:
        return None, None, None, None, None, None, None

    metric_col = find_metric_column_for_country(year_df, country, metric_key)
    if not metric_col:
        return None, None, None, None, None, None, None

    year_df = ensure_product_column(year_df)
    year_df[metric_col] = pd.to_numeric(year_df[metric_col], errors='coerce')
    year_df['PRODUCTNAME'] = year_df['PRODUCTNAME'].astype(str)

    # Sort product bins: numeric first by value, then non-numeric
    products = sorted(
        year_df['PRODUCTNAME'].dropna().unique().tolist(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))
    )
    if not products:
        return None, None, None, None, None, None, None

    grouped = (
        year_df
        .dropna(subset=[metric_col])
        .groupby(['MONTH_NAME', 'PRODUCTNAME'])[metric_col]
        .mean()
        .reset_index()
    )

    months_label = [month_abbr[m] for m in range(1, 13)]

    # Build full grid of months x products to keep order
    all_months = pd.DataFrame({'MONTH_NAME': months_label})
    all_prods = pd.DataFrame({'PRODUCTNAME': products})
    all_months['k'] = 1
    all_prods['k'] = 1
    full_index = pd.merge(all_months, all_prods, on='k').drop(columns='k')

    merged = pd.merge(full_index, grouped, on=['MONTH_NAME', 'PRODUCTNAME'], how='left')
    heatmap = merged.pivot(index='MONTH_NAME', columns='PRODUCTNAME', values=metric_col)
    heatmap = heatmap.reindex(index=months_label, columns=products)

    x_labels_bins = [product_bin_label(p) for p in products]

    spec = METRICS[metric_key]
    cbar_label = spec["cbar_label"]
    cmap = spec["cmap"]
    center = spec["center"]
    title_suffix = spec["title_suffix"]

    return heatmap, x_labels_bins, months_label, cbar_label, cmap, center, title_suffix

# ---------------- UI ----------------
st.title("FCR Heatmap — Price, Demand, Import/Export")
st.caption("Reads local Excel files named: RESULT_OVERVIEW_CAPACITY_MARKET_FCR_YYYY.xlsx")

# Sidebar controls
with st.sidebar:
    # Year
    year_default_index = len(YEARS) - 1 if YEARS else 0
    year = st.selectbox("Year", YEARS, index=year_default_index)

    # Load file for year
    path = find_local_file_for_year(year)
    if not path or not os.path.exists(path):
        st.error(
            f"File not found for {year}. Expected name: "
            f"`{FILENAME_PATTERN.format(y=year)}` in the app folder or `./data/`."
        )
        st.stop()

    mtime = os.path.getmtime(path)
    with st.spinner(f"Loading {os.path.basename(path)} …"):
        df_year = load_year_df(path, mtime)

    if df_year is None:
        st.error("Could not load or parse the Excel file (missing 'DATE_FROM' or empty).")
        st.stop()

    # Countries (from any metric)
    countries = extract_countries_from_df(df_year)
    if not countries:
        st.error("No countries detected in the file. Check the column names.")
        st.stop()

    # Try to default to BELGIUM if present; else first in the list
    if "BELGIUM" in countries:
        default_country_idx = countries.index("BELGIUM")
    else:
        default_country_idx = 0

    country = st.selectbox("Country", countries, index=default_country_idx)

    # Metric selection
    metric_options = {
        "PRICE": "Settlement Capacity Price (€/MW)",
        "DEMAND": "Demand (MW)",
        "IMPORT_EXPORT": "Import (−) / Export (+) (MW)"
    }
    metric_key = st.selectbox(
        "Metric",
        list(metric_options.keys()),
        format_func=lambda k: metric_options[k],
        index=0
    )

# Main visualization
heatmap_data, x_labels_bins, months_label, cbar_label, cmap, center, title_suffix = build_heatmap_for(
    df_year, year, country, metric_key
)

if heatmap_data is None or heatmap_data.empty:
    st.warning("No data found for this selection (metric & country).")
else:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.set(style="white")

    # Choose diverging for import/export with center=0
    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap=cmap,
        center=center,
        cbar_kws={'label': cbar_label},
        ax=ax
    )

    spec = METRICS[metric_key]
    ax.set_title(f"{spec['title_suffix']} — {country} — {year}")
    ax.set_xticks([i + 0.5 for i in range(len(heatmap_data.columns))])
    ax.set_xticklabels(x_labels_bins, rotation=45, ha='right')
    ax.set_yticks([i + 0.5 for i in range(len(heatmap_data.index))])
    ax.set_yticklabels(months_label, rotation=0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig)

# Notes
st.markdown(
    """
**Notes**

- Place files next to `app.py` or under `./data/`.
- File name must be exactly `RESULT_OVERVIEW_CAPACITY_MARKET_FCR_YYYY.xlsx`.
- Country-specific columns follow these patterns (prefix is the country code, e.g., `AT`, `BE`, …):
  - **Price**: `CC_SETTLEMENTCAPACITY_PRICE_[EUR/MW]`
  - **Demand**: `CC_DEMAND_[MW]`
  - **Import(−)/Export(+)**: `CC_IMPORT(-)_EXPORT(+)_[MW]`
- The heatmap shows monthly **average values** per `PRODUCTNAME`.  
  If `PRODUCTNAME` is missing in your file, the app will aggregate into a single bucket **ALL**.
- Import(−)/Export(+) uses a diverging colormap centered at 0:
  - **Blue** = Import (negative)
  - **Red** = Export (positive)
"""
)
