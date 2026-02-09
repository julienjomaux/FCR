
import os
import glob
from typing import Optional, List, Dict
from datetime import date

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from calendar import month_abbr

# ---------------- Page setup ----------------
st.set_page_config(page_title="FCR — Price Heatmap", layout="wide", page_icon="GEM.webp")

# Years you want to expose in the UI (adapt if you add more files)
YEARS = [2021, 2022, 2023, 2024, 2025]

# Where to look for the Excel files.
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

NON_COUNTRIES = {"CROSSBORDER", "CROSS-BORDER"}  # filtered out

def harmonize_country(name: str) -> str:
    name = str(name).upper().strip()
    return COUNTRY_RENAME.get(name, name)

def product_bin_label(product: str) -> str:
    try:
        val = int(str(product).strip())
        return f"{val} to {val+4}"
    except Exception:
        return str(product)

def find_local_file_for_year(year: int) -> Optional[str]:
    candidates = []
    for loc in SEARCH_LOCATIONS:
        pattern = os.path.join(loc, FILENAME_PATTERN.format(y=year))
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return os.path.abspath(candidates[0])

@st.cache_data(show_spinner=False)
def load_year_df(path: str, mtime: float) -> Optional[pd.DataFrame]:
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
METRICS: Dict[str, Dict] = {
    "PRICE": {
        "label": "Settlement Capacity Price",
        "suffixes": ["SETTLEMENTCAPACITY_PRICE_[EUR/MW]"],
        "unit": "€/MW",
        "cmap": "YlOrRd",
        "center": None,
        "title_suffix": "Average Capacity Price FCR",
    },
    "DEMAND": {
        "label": "Demand",
        "suffixes": ["DEMAND_[MW]"],
        "unit": "MW",
    },
    "IMPORT_EXPORT": {
        "label": "Import (−) / Export (+)",
        "suffixes": [
            "IMPORT(-)_EXPORT(+)_[MW]",
            "DEFICIT(-)_SURPLUS(+)_[MW]",
        ],
        "unit": "MW",
        "cmap": "coolwarm",
        "center": 0.0,
        "title_suffix": "Average Import(−)/Export(+) FCR",
    },
}

def extract_countries_from_df(df: pd.DataFrame) -> List[str]:
    all_suffixes: List[str] = []
    for spec in METRICS.values():
        all_suffixes.extend(spec["suffixes"])

    candidates = set()
    for col in df.columns:
        col_str = str(col)
        for suf in all_suffixes:
            if col_str.endswith(suf):
                prefix = col_str.split('_')[0]
                label = harmonize_country(prefix)
                if label not in NON_COUNTRIES:
                    candidates.add(label)
                break
    return sorted(candidates)

def find_metric_column_for_country(df: pd.DataFrame, country: str, metric_key: str) -> Optional[str]:
    suffixes = METRICS[metric_key]["suffixes"]
    matches = []
    for col in df.columns:
        col_str = str(col)
        for suf in suffixes:
            if col_str.endswith(suf):
                prefix = col_str.split('_')[0]
                if harmonize_country(prefix) == country:
                    matches.append(col_str)
                    break
    if matches and metric_key == "IMPORT_EXPORT":
        def pref_score(c):
            return 0 if c.endswith("IMPORT(-)_EXPORT(+)_[MW]") else 1
        matches.sort(key=pref_score)
    return matches[0] if matches else None

def ensure_product_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'PRODUCTNAME' not in df.columns:
        df['PRODUCTNAME'] = 'ALL'
    return df

def build_heatmap_for(df: pd.DataFrame, year: int, country: str, metric_key: str):
    year_df = df[df['YEAR'] == year].copy()
    if year_df.empty:
        return None, None, None, None, None, None, None

    metric_col = find_metric_column_for_country(year_df, country, metric_key)
    if not metric_col:
        return None, None, None, None, None, None, None

    year_df = ensure_product_column(year_df)
    year_df[metric_col] = pd.to_numeric(year_df[metric_col], errors='coerce')
    year_df['PRODUCTNAME'] = year_df['PRODUCTNAME'].astype(str)

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

    all_months = pd.DataFrame({'MONTH_NAME': months_label})
    all_prods = pd.DataFrame({'PRODUCTNAME': products})
    all_months['k'] = 1
    all_prods['k'] = 1
    full_index = pd.merge(all_months, all_prods, on='k').drop(columns='k')

    merged = pd.merge(full_index, grouped, on=['MONTH_NAME', 'PRODUCTNAME'], how='left')
    heatmap = merged.pivot(index='MONTH_NAME', columns='PRODUCTNAME', values=metric_col)
    heatmap = heatmap.reindex(index=months_label, columns=products)

    x_labels_bins = ["0 to 4", "4 to 8", "8 to 12", "12 to 16", "16 to 20", "20 to 24"]

    spec = METRICS[metric_key]
    return heatmap, x_labels_bins, months_label, spec["unit"], spec.get("cmap"), spec.get("center"), spec["title_suffix"]

def collect_demand_columns(df: pd.DataFrame) -> Dict[str, str]:
    suffixes = METRICS["DEMAND"]["suffixes"]
    mapping = {}
    for col in df.columns:
        col_str = str(col)
        for suf in suffixes:
            if col_str.endswith(suf):
                prefix = col_str.split('_')[0]
                cname = harmonize_country(prefix)
                if cname not in NON_COUNTRIES:
                    mapping[cname] = col_str
                break
    return mapping

def demand_bar_data(df: pd.DataFrame, year: int) -> Optional[pd.DataFrame]:
    year_df = df[df['YEAR'] == year].copy()
    if year_df.empty:
        return None

    mapping = collect_demand_columns(year_df)
    if not mapping:
        return None

    rows = []
    for cname, col in mapping.items():
        vals = pd.to_numeric(year_df[col], errors='coerce')
        if vals.notna().any():
            rows.append({"Country": cname, "Demand (MW)": vals.mean()})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("Demand (MW)", ascending=False)

def specific_day_bar_data(df: pd.DataFrame, the_date: date, country: str, metric_key: str):
    day_df = df[df['DATE'].dt.date == the_date]
    if day_df.empty:
        return None

    metric_col = find_metric_column_for_country(day_df, country, metric_key)
    if not metric_col:
        return None

    day_df = ensure_product_column(day_df)
    day_df[metric_col] = pd.to_numeric(day_df[metric_col], errors='coerce')
    day_df['PRODUCTNAME'] = day_df['PRODUCTNAME'].astype(str)

    grouped = (
        day_df.dropna(subset=[metric_col])
        .groupby('PRODUCTNAME')[metric_col]
        .mean()
        .reset_index()
    )
    if grouped.empty:
        return None

    grouped['__k'] = grouped['PRODUCTNAME'].apply(
        lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))
    )
    grouped = grouped.sort_values('__k').drop(columns='__k')

    grouped['Product'] = grouped['PRODUCTNAME'].apply(product_bin_label)
    grouped = grouped.rename(columns={metric_col: 'Value'})[['Product', 'Value']]
    return grouped

# ---------------- TOP TEXT (kept exactly as you wrote) ----------------
st.markdown(
    """
This app presents heatmaps and daily views of FCR from 201 to 2025. 

- **Heatmaps:** Show monthly average per product with the sidebar. 
- **Daily view:** Displays capacity prices for all the blocks. 

You can change the considered country and the year easily. 

**Data source:** Regelleistung.net

**More insights:** [GEM Energy Analytics](https://gemenergyanalytics.substack.com/)  
**Connect with me:** Julien Jomaux  
**Email me:** julien.jomaux@gmail.com

If you want to support, please consider becoming a paying member of GEM Energy Analytics. 

Thanks for reading [GEM Energy Analytics](https://gemenergyanalytics.substack.com/).
"""
)

# ---------------- UI ----------------
st.title("FCR — Price Heatmap")

with st.sidebar:
    year_default_index = len(YEARS) - 1 if YEARS else 0
    year = st.selectbox("Year", YEARS, index=year_default_index)

    path = find_local_file_for_year(year)
    if not path or not os.path.exists(path):
        st.error(
            f"File not found for {year}. Expected name: "
            f"`{FILENAME_PATTERN.format(y=year)}` in app folder or ./data/."
        )
        st.stop()

    mtime = os.path.getmtime(path)
    with st.spinner(f"Loading {os.path.basename(path)} …"):
        df_year = load_year_df(path, mtime)
    if df_year is None:
        st.error("Could not load Excel file.")
        st.stop()

    countries = extract_countries_from_df(df_year)
    if not countries:
        st.error("No countries detected.")
        st.stop()

    default_country_idx = countries.index("BELGIUM") if "BELGIUM" in countries else 0
    country = st.selectbox("Country", countries, index=default_country_idx)

    metric_options = {
        "PRICE": "Settlement Capacity Price (€/MW)",
        "IMPORT_EXPORT": "Import (−) / Export (+) (MW)",
    }
    metric_key = st.selectbox(
        "Metric",
        list(metric_options.keys()),
        format_func=lambda k: metric_options[k],
        index=0,
    )

    min_d = pd.to_datetime(df_year['DATE'].min()).date()
    max_d = pd.to_datetime(df_year['DATE'].max()).date()
    chosen_day = st.date_input(
        "Specific date (for day bar chart)",
        value=min_d,
        min_value=min_d,
        max_value=max_d
    )

# ---------------- CONTENT (now always visible) ----------------

# 1) HEATMAP
heatmap_data, x_labels_bins, months_label, unit, cmap, center, title_suffix = build_heatmap_for(
    df_year, year, country, metric_key
)

st.subheader(f"{title_suffix} — {country} — {year}")
if heatmap_data is None or heatmap_data.empty:
    st.warning("No data found.")
else:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.set(style="white")
    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap=cmap,
        center=center,
        cbar_kws={'label': unit},
        ax=ax
    )

    ax.text(
        0.5, 0.5,
        "gemenergyanalytics.substack.com\nJulien Jomaux",
        color='gray',
        fontsize=32,
        alpha=0.3,
        ha='center',
        va='center',
        rotation=30,
        transform=ax.transAxes,
    )

    ax.set_xticks([i + 0.5 for i in range(len(heatmap_data.columns))])
    ax.set_xticklabels(x_labels_bins, rotation=45, ha='right')
    ax.set_yticks([i + 0.5 for i in range(len(heatmap_data.index))])
    ax.set_yticklabels(months_label)
    plt.tight_layout()
    st.pyplot(fig)

# 2) DEMAND (bar chart)
st.subheader(f"Demand per country — {year}")
demand_df = demand_bar_data(df_year, year)
if demand_df is None:
    st.info("No Demand data found.")
else:
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    sns.set(style="whitegrid")
    sns.barplot(
        data=demand_df,
        x="Country",
        y="Demand (MW)",
        palette="Blues_d",
        ax=ax2
    )
    ax2.set_title(f"Average Demand (MW) — {year}")
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

# 3) SPECIFIC-DAY VIEW
metric_label = METRICS[metric_key]["label"]
st.subheader(f"{metric_label} — {country} — {chosen_day.isoformat()}")
st.markdown("This presents the results of the auction per day.")

day_df = specific_day_bar_data(df_year, chosen_day, country, metric_key)
if day_df is None:
    st.info("No data for this date.")
else:
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    sns.set(style="whitegrid")
    sns.barplot(
        data=day_df,
        x="Product",
        y="Value",
        color="#4472C4",
        ax=ax3
    )
    ax3.set_ylabel("€/MW/block")
    ax3.set_title(f"{metric_label} — {country} — {chosen_day.isoformat()}")
    ax3.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)

