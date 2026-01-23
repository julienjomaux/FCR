
# app.py
import os
import io
import glob
import time
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from calendar import month_abbr

# ---------------- Page setup ----------------
st.set_page_config(page_title="FCR Capacity Price Heatmap", layout="wide")

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

def find_local_file_for_year(year: int) -> str | None:
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
def load_year_df(path: str, mtime: float) -> pd.DataFrame | None:
    """
    Load the given Excel file and return a cleaned DataFrame.
    Cache is keyed by (path, mtime) via arguments.
    """
    try:
        # openpyxl is the default engine for .xlsx in pandas >= 2, but being explicit is fine
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

def extract_countries_from_df(df: pd.DataFrame) -> list[str]:
    """Detect available countries based on *_SETTLEMENTCAPACITY_PRICE_[EUR/MW] columns."""
    candidates = set()
    for col in df.columns:
        col = str(col)
        if col.endswith('SETTLEMENTCAPACITY_PRICE_[EUR/MW]'):
            code = col.split('_')[0]
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
    """
    Returns (heatmap_data, x_labels_bins, months_label)
    - heatmap_data: index: months (Jan..Dec), columns: PRODUCTNAME (sorted)
    - x_labels_bins: pretty x labels (e.g., '0 to 4')
    - months_label: ['Jan', ..., 'Dec']
    """
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

    grouped = (year_df
               .dropna(subset=[price_col])
               .groupby(['MONTH_NAME', 'PRODUCTNAME'])[price_col]
               .mean()
               .reset_index())

    months_label = [month_abbr[m] for m in range(1, 13)]
    all_months = pd.DataFrame({'MONTH_NAME': months_label})
    all_prods = pd.DataFrame({'PRODUCTNAME': products})
    all_months['k'] = 1
    all_prods['k'] = 1
    full_index = pd.merge(all_months, all_prods, on='k').drop(columns='k')

    merged = pd.merge(full_index, grouped, on=['MONTH_NAME', 'PRODUCTNAME'], how='left')
    heatmap = merged.pivot(index='MONTH_NAME', columns='PRODUCTNAME', values=price_col)
    heatmap = heatmap.reindex(index=months_label, columns=products)

    x_labels_bins = [product_bin_label(p) for p in products]
    return heatmap, x_labels_bins, months_label

# ---------------- UI ----------------
st.title("FCR Capacity Price Heatmap (€/MW per block)")
st.caption("Reads local Excel files named: RESULT_OVERVIEW_CAPACITY_MARKET_FCR_YYYY.xlsx")

left, right = st.columns([1, 3])

with left:
    year = st.selectbox("Select year", YEARS, index=len(YEARS)-1)

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

    countries = extract_countries_from_df(df_year)
    if not countries:
        st.error("No countries detected in the file. Check the column names.")
        st.stop()

    country = st.selectbox("Select country", countries)

with right:
    heatmap_data, x_labels_bins, months_label = build_heatmap_for(df_year, year, country)

    if heatmap_data is None or heatmap_data.empty:
        st.warning("No price data found for this selection.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            heatmap_data,
            annot=False,
            cmap="YlOrRd",
            cbar_kws={'label': '€/MW'},
            ax=ax
        )
        ax.set_title(f"Average Capacity Price FCR — {country} — {year}")
        ax.set_xticks([i + 0.5 for i in range(len(heatmap_data.columns))])
        ax.set_xticklabels(x_labels_bins, rotation=45, ha='right')
        ax.set_yticks([i + 0.5 for i in range(len(heatmap_data.index))])
        ax.set_yticklabels(months_label, rotation=0)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig)

st.markdown(
    """
**Notes**
- Place files next to `app.py` or under `./data/`.
- File name must be exactly `RESULT_OVERVIEW_CAPACITY_MARKET_FCR_YYYY.xlsx`.
- The heatmap shows monthly **average settlement capacity prices** (€/MW) per `PRODUCTNAME`.
"""
)
