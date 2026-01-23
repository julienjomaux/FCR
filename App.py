
# app.py
import io
import requests
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from calendar import month_abbr

# -------------- Page & constants --------------
st.set_page_config(page_title="FCR Capacity Price Heatmap", layout="wide")

YEARS = [2021, 2022, 2023, 2024, 2025]
BASE_URL = (
    "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/"
    "RESULT_OVERVIEW_CAPACITY_MARKET_FCR_{y}-01-01_{y}-12-31.xlsx"
)

# Expanded harmonization map (extend as needed)
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
    # Turns "0" -> "0 to 4", "5" -> "5 to 9" etc., otherwise returns as-is
    try:
        val = int(product)
        return f"{val} to {val+4}"
    except Exception:
        return str(product)

# -------------- Data layer (cached) --------------
@st.cache_data(show_spinner=False)
def download_year_df(year: int) -> pd.DataFrame | None:
    """Downloads the Excel for a given year and returns a cleaned DataFrame, or None if unavailable."""
    url = BASE_URL.format(y=year)
    try:
        # verify=True should be OK; fallback to False if needed.
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200 or len(resp.content) < 10000:
            # Retry with verify=False if the server insists (rare)
            resp = requests.get(url, timeout=60, verify=False)
            if resp.status_code != 200:
                return None
        df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
    except Exception:
        return None

    if df is None or df.empty or 'DATE_FROM' not in df.columns:
        return None

    # Basic date fields
    df['DATE'] = pd.to_datetime(df['DATE_FROM'], dayfirst=True, errors='coerce')
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month
    df['MONTH_NAME'] = df['DATE'].dt.strftime('%b')

    return df

def extract_countries_from_df(df: pd.DataFrame) -> list[str]:
    """Detects available countries in a given year's file based on the price column pattern."""
    candidates = set()
    for col in df.columns:
        if str(col).endswith('SETTLEMENTCAPACITY_PRICE_[EUR/MW]'):
            code = str(col).split('_')[0]
            candidates.add(harmonize_country(code))
    return sorted(candidates)

def find_price_column_for_country(df: pd.DataFrame, country: str) -> str | None:
    """
    In the provided DataFrame, find the price column associated to a given country.
    Columns are like: '{CODE}_SETTLEMENTCAPACITY_PRICE_[EUR/MW]'.
    We harmonize the prefix and match to the `country` label.
    """
    keys = []
    for col in df.columns:
        col_str = str(col)
        if col_str.endswith('SETTLEMENTCAPACITY_PRICE_[EUR/MW]'):
            prefix = col_str.split('_')[0]
            if harmonize_country(prefix) == country:
                keys.append(col_str)
    # Expect one; if multiple, take the first
    return keys[0] if keys else None

def build_heatmap_for(df: pd.DataFrame, year: int, country: str):
    """
    Returns (heatmap_data DataFrame, x_labels_bins, y_month_labels) for plotting.
    heatmap_data has index=month names (Jan..Dec) and columns=PRODUCTNAME sorted.
    """
    # Year subset
    year_df = df[df['YEAR'] == year].copy()
    if year_df.empty:
        return None, None, None

    price_col = find_price_column_for_country(year_df, country)
    if not price_col:
        return None, None, None

    # Clean price & product
    year_df[price_col] = pd.to_numeric(year_df[price_col], errors='coerce')
    year_df['PRODUCTNAME'] = year_df['PRODUCTNAME'].astype(str)

    # Available product blocks
    products = sorted(
        year_df['PRODUCTNAME'].dropna().unique().tolist(),
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))
    )
    if not products:
        return None, None, None

    # Monthly average per product
    g = (year_df
         .dropna(subset=[price_col])
         .groupby(['MONTH_NAME', 'PRODUCTNAME'])[price_col]
         .mean()
         .reset_index())

    # Ensure full 12 months x all products grid
    months_label = [month_abbr[m] for m in range(1, 13)]
    all_months = pd.DataFrame({'MONTH_NAME': months_label})
    all_products = pd.DataFrame({'PRODUCTNAME': products})
    all_months['__k'] = 1
    all_products['__k'] = 1
    full_index = pd.merge(all_months, all_products, on='__k', how='outer').drop(columns='__k')

    merged = pd.merge(full_index, g, on=['MONTH_NAME', 'PRODUCTNAME'], how='left')
    heatmap_data = merged.pivot(index='MONTH_NAME', columns='PRODUCTNAME', values=price_col)
    heatmap_data = heatmap_data.reindex(index=months_label, columns=products)

    x_labels_bins = [product_bin_label(p) for p in products]
    return heatmap_data, x_labels_bins, months_label

# -------------- UI --------------
st.title("FCR Capacity Price Heatmap (€/MW per block)")
st.caption("Source: regelleistung.net – FCR Capacity Market Results")

col_left, col_right = st.columns([1, 3])

with col_left:
    year = st.selectbox("Select year", YEARS, index=len(YEARS)-1)
    with st.spinner(f"Downloading data for {year}..."):
        df_year = download_year_df(year)

    if df_year is None:
        st.error("No data available for this year (download failed or format changed). Try another year.")
        st.stop()

    countries = extract_countries_from_df(df_year)
    if not countries:
        st.error("No countries detected in this file. The structure may have changed.")
        st.stop()

    country = st.selectbox("Select country", countries)

with col_right:
    heatmap_data, x_labels_bins, months_label = build_heatmap_for(df_year, year, country)

    if heatmap_data is None or heatmap_data.empty:
        st.warning("No price data found for this selection.")
    else:
        # Plot
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
    - The heatmap shows monthly average settlement capacity prices (€/MW) for each product block in the selected year & country.
    - Product labels are displayed as 5-minute blocks (e.g., "0 to 4", "5 to 9") when numeric; non-numeric labels are preserved as-is.
    """
)
