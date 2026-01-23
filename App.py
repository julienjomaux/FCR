
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
st.set_page_config(page_title="FCR — Price Heatmap, Demand & Day View", layout="wide")

# Years you want to expose in the UI (adapt if you add more files)
YEARS = [2021, 2022, 2023, 2024, 2025]

# Where to look for the Excel files.
# The app will first try the current folder, then in ./data subfolder.
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

NON_COUNTRIES = {"CROSSBORDER", "CROSS-BORDER"}  # filtered out from lists/charts

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
# Note: suffixes can be a list to support legacy/alternative names.
METRICS: Dict[str, Dict] = {
    "PRICE": {
        "label": "Settlement Capacity Price",
        "suffixes": ["SETTLEMENTCAPACITY_PRICE_[EUR/MW]"],
        "unit": "€/MW",
        "cmap": "YlOrRd",
        "center": None,
        "title_suffix": "Average Capacity Price FCR",
    },
    # Demand is visualized separately (bar chart across countries)
    "DEMAND": {
        "label": "Demand",
        "suffixes": ["DEMAND_[MW]"],
        "unit": "MW",
    },
    "IMPORT_EXPORT": {
        "label": "Import (−) / Export (+)",
        # Support both naming schemes:
        #   CC_IMPORT(-)_EXPORT(+)_[MW]
        #   CC_DEFICIT(-)_SURPLUS(+)_[MW]
        "suffixes": [
            "IMPORT(-)_EXPORT(+)_[MW]",
            "DEFICIT(-)_SURPLUS(+)_[MW]",
        ],
        "unit": "MW",
        "cmap": "coolwarm",
        "center": 0.0,  # diverging, centered at 0
        "title_suffix": "Average Import(−)/Export(+) FCR",
    },
}

def extract_countries_from_df(df: pd.DataFrame) -> List[str]:
    """
    Detect available countries based on any of the metric suffixes.
    Filter out non-country labels like CROSSBORDER.
    """
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
    """
    From df, find the column name matching the selected country and metric.
    Accept any of the possible suffix variants for that metric.
    Country is tested via the prefix before the first underscore.
    """
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
    # Prefer deterministic order (prefer IMPORT/EXPORT over DEFICIT/SURPLUS if both exist)
    if matches and metric_key == "IMPORT_EXPORT":
        def pref_score(cname: str) -> int:
            return 0 if cname.endswith("IMPORT(-)_EXPORT(+)_[MW]") else 1
        matches.sort(key=pref_score)
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
    Returns (heatmap_data, x_labels_bins, months_label, unit, cmap, center, title_suffix)
    - heatmap_data: index: months (Jan..Dec), columns: PRODUCTNAME (sorted)
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

    # Full grid of months x products to keep order
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
    unit = spec["unit"]
    cmap = spec.get("cmap")
    center = spec.get("center")
    title_suffix = spec["title_suffix"]

    return heatmap, x_labels_bins, months_label, unit, cmap, center, title_suffix

def collect_demand_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Return a mapping of country_name -> DEMAND column for that country.
    Filters out non-country labels (e.g., CROSSBORDER).
    """
    suffixes = METRICS["DEMAND"]["suffixes"]
    mapping: Dict[str, str] = {}
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
    """
    Build a dataframe with average demand per country for the selected year.
    Returns columns: ['Country', 'Demand (MW)']
    """
    year_df = df[df['YEAR'] == year].copy()
    if year_df.empty:
        return None

    mapping = collect_demand_columns(year_df)
    if not mapping:
        return None

    out_rows = []
    for cname, col in mapping.items():
        vals = pd.to_numeric(year_df[col], errors='coerce')
        if vals.notna().any():
            out_rows.append({"Country": cname, "Demand (MW)": vals.mean()})
    if not out_rows:
        return None
    out = pd.DataFrame(out_rows).sort_values("Demand (MW)", ascending=False)
    return out

def specific_day_bar_data(df: pd.DataFrame, the_date: date, country: str, metric_key: str) -> Optional[pd.DataFrame]:
    """
    Build a dataframe with daily mean per product for given date/country/metric.
    Returns columns: ['Product', 'Value']
    """
    day_df = df.copy()
    day_df = day_df[day_df['DATE'].dt.date == the_date]
    if day_df.empty:
        return None

    metric_col = find_metric_column_for_country(day_df, country, metric_key)
    if not metric_col:
        return None

    day_df = ensure_product_column(day_df)
    day_df[metric_col] = pd.to_numeric(day_df[metric_col], errors='coerce')
    day_df['PRODUCTNAME'] = day_df['PRODUCTNAME'].astype(str)

    grouped = (
        day_df
        .dropna(subset=[metric_col])
        .groupby('PRODUCTNAME')[metric_col]
        .mean()  # daily mean per product (not monthly)
        .reset_index()
    )
    if grouped.empty:
        return None

    # Order products as before (numeric bins first)
    grouped['__order_key__'] = grouped['PRODUCTNAME'].apply(
        lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))
    )
    grouped = grouped.sort_values('__order_key__').drop(columns='__order_key__')

    grouped['Product'] = grouped['PRODUCTNAME'].apply(product_bin_label)
    grouped = grouped.rename(columns={metric_col: 'Value'})[['Product', 'Value']]
    return grouped

# ---------- Safe config getter (supports python-decouple and env vars) ----------
def get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    # Try python-decouple if available; fallback to environment variables
    try:
        from decouple import config as decouple_config  # type: ignore
        return decouple_config(key, default=default)
    except Exception:
        return os.getenv(key, default)

# ---------------- Top: Sign-up / Login section ----------------
stripe_link = get_config_value('STRIPE_CHECKOUT_LINK', '#')
secret_password = get_config_value('SECRET_PASSWORD', '')

# --- Signup callout with hyperlink to secret STRIPE_CHECKOUT_LINK ---
st.markdown(
    f"""
    If you want to access all the apps of GEM Energy Analytics, please sign up following the link below. 

    Currently, the fee is 30 € per month. When the payment is done, you will receive an password that will grant you access to all apps. Every month, you will receive an email with a new password to access the apps (except if you unsubscribe). 
    Feel free to reach out at Julien.jomaux@gmail.com

    #### Sign Up Now :metal:
    """
)

with st.form("login_form"):
    st.write("Login")
    # Email removed as requested; password only
    password = st.text_input('Enter Your Password', type="password")
    submitted = st.form_submit_button("Login")

if submitted:
    if secret_password and (password == secret_password):
        st.session_state['logged_in'] = True
        st.success('Successfully Logged In!')
    else:
        st.session_state['logged_in'] = False
        st.error('Incorrect login credentials.')

# ---------------- UI ----------------
st.title("FCR — Price Heatmap, Demand (All Countries), and Specific-Day View")
st.caption("Reads local Excel files named: RESULT_OVERVIEW_CAPACITY_MARKET_FCR_YYYY.xlsx")

# Sidebar controls (available regardless), visualizations are gated below
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

    # Countries (from any metric), excluding non-countries
    countries = extract_countries_from_df(df_year)
    if not countries:
        st.error("No countries detected in the file. Check the column names.")
        st.stop()

    # Try to default to BELGIUM if present; else first in the list
    default_country_idx = countries.index("BELGIUM") if "BELGIUM" in countries else 0
    country = st.selectbox("Country", countries, index=default_country_idx)

    # Metric selection — ONLY Price or Import/Export (Demand shown separately as bar chart)
    metric_options = {
        "PRICE": "Settlement Capacity Price (€/MW)",
        "IMPORT_EXPORT": "Import (−) / Export (+) (MW)",
    }
    metric_key = st.selectbox(
        "Metric",
        list(metric_options.keys()),
        format_func=lambda k: metric_options[k],
        index=0
    )

    # Specific date for the third chart
    # Limit the date picker to the available dates in this year's file
    min_d = pd.to_datetime(df_year['DATE'].min()).date() if not df_year.empty and pd.notna(df_year['DATE'].min()) else date(year, 1, 1)
    max_d = pd.to_datetime(df_year['DATE'].max()).date() if not df_year.empty and pd.notna(df_year['DATE'].max()) else date(year, 12, 31)
    default_day = min(max_d, max(min_d, date(year, 1, 1)))
    chosen_day = st.date_input(
        "Specific date (for day bar chart)",
        value=default_day,
        min_value=min_d,
        max_value=max_d
    )

# --------------- GATED CONTENT: only visible after successful login ---------------
is_logged_in = st.session_state.get('logged_in', False)

if not is_logged_in:
    st.info("🔒 Please log in with the password above to access the charts.")
else:
    # ---------------- 1) Main heatmap (Price or Import/Export) ----------------
    with st.container():
        heatmap_data, x_labels_bins, months_label, unit, cmap, center, title_suffix = build_heatmap_for(
            df_year, year, country, metric_key
        )

        st.subheader(f"{title_suffix} — {country} — {year}")
        if heatmap_data is None or heatmap_data.empty:
            st.warning("No data found for this selection (metric & country).")
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

            # ----- Watermark banner on the heatmap (two lines, slightly smaller) -----
            ax.text(
                0.5, 0.5,
                "gemenergyanalytics.substack.com\nJulien Jomaux",
                color='gray',
                fontsize=32,      # slightly smaller than before (was 40)
                alpha=0.3,
                ha='center',
                va='center',
                rotation=30,
                transform=ax.transAxes,
                zorder=1
            )

            ax.set_xticks([i + 0.5 for i in range(len(heatmap_data.columns))])
            ax.set_xticklabels(x_labels_bins, rotation=45, ha='right')
            ax.set_yticks([i + 0.5 for i in range(len(heatmap_data.index))])
            ax.set_yticklabels(months_label, rotation=0)
            ax.set_xlabel('')
            ax.set_ylabel('')
            plt.tight_layout()
            st.pyplot(fig)

    # ---------------- 2) Demand bar chart (all countries together) ----------------
    with st.container():
        st.subheader(f"Demand per country — Yearly average — {year}")
        demand_df = demand_bar_data(df_year, year)
        if demand_df is None or demand_df.empty:
            st.info("No Demand data found for this year.")
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
            ax2.set_xlabel("")
            ax2.set_ylabel("Demand (MW)")
            ax2.set_title(f"Average Demand (MW) — {year}")
            ax2.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)

    # ---------------- 3) Specific-date bar chart (Price or Import/Export) ----------------
    with st.container():
        metric_label = METRICS[metric_key]["label"]
        unit_label = METRICS[metric_key]["unit"]
        st.subheader(f"{metric_label} — {country} — {chosen_day.isoformat()} (daily mean per product)")
        day_df = specific_day_bar_data(df_year, chosen_day, country, metric_key)
        if day_df is None or day_df.empty:
            st.info("No data for the selected date/country/metric.")
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
            ax3.set_xlabel("")
            ax3.set_ylabel(unit_label)
            ax3.set_title(f"{metric_label} — {country} — {chosen_day.isoformat()}")
            ax3.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig3)

    # Notes (also gated)
    st.markdown(
        """
    **Notes**

    - Place files next to `app.py` or under `./data/`.
    - File name must be exactly `RESULT_OVERVIEW_CAPACITY_MARKET_FCR_YYYY.xlsx`.
    - Country-specific columns follow these patterns (prefix is the country code or full name, e.g., `AT`, `AUSTRIA`, `BE`, …):
      - **Price**: `CC_SETTLEMENTCAPACITY_PRICE_[EUR/MW]`
      - **Import(−)/Export(+)** (both supported):
        - `CC_IMPORT(-)_EXPORT(+)_[MW]`
        - `CC_DEFICIT(-)_SURPLUS(+)_[MW]`
      - **Demand** (for the all-countries bar chart): `CC_DEMAND_[MW]`
    - **CROSSBORDER**/**CROSS-BORDER** is excluded from countries and from the Demand chart.
    - Heatmap shows **monthly averages** by `PRODUCTNAME`.
    - Specific-day bar chart shows **daily mean per product** (not monthly averages).  
      If you prefer sum or a specific intraday slice, let me know.
    """
    )
