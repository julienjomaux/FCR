import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.title("FCR Capacity Market Results Viewer")

# The file URL
url = "https://www.regelleistung.net/apps/cpp-publisher/api/v2/tenders/files/RESULT_OVERVIEW_CAPACITY_MARKET_FCR_2025-01-01_2025-12-31.xlsx"

st.write("Downloading the Excel file...")

# Download the file
response = requests.get(url)
if response.status_code == 200:
    # Load into pandas DataFrame
    file_bytes = BytesIO(response.content)
    df = pd.read_excel(file_bytes)
    st.write("First 10 rows of the file:")
    st.dataframe(df.head(10))
else:
    st.error("Failed to download file. HTTP status code: {}".format(response.status_code))
