# This code of the Streamlit app visualizes data from the following dataset:
# https://app.snowflake.com/marketplace/listing/GZSVZ8ON6J/dataconsulting-pl-opencellid-open-database-of-cell-towers

import streamlit as st
import h3
import pandas as pd
import pydeck as pdk
import branca.colormap as cm
from PIL import Image

DATA_PATH = "/data/cell_towers.csv"  # uploaded dataset

@st.cache_data(ttl=60 * 60 * 24 * 2)  # cache for 2 days
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # normalize common column names just in case
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # required columns: lat, lon, mcc
    if not {"lat", "lon", "mcc"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: lat, lon, mcc")
    return df

@st.cache_data(ttl=60 * 60 * 24 * 2)
def get_h3_df(resolution: int) -> pd.DataFrame:
    df = load_data().query("310 <= mcc <= 316").copy()
    # compute H3 index for each point (resolution from the slider)
    df["H3"] = df.apply(lambda r: _latlng_to_h3(float(r["lat"]), float(r["lon"]), int(resolution)), axis=1)
    # aggregate counts per hex
    out = df.groupby("H3").size().reset_index(name="COUNT")
    return out

@st.cache_data(ttl=60 * 60 * 24 * 2)
def get_h3_layer(df: pd.DataFrame) -> pdk.Layer:
    return pdk.Layer(
        "H3HexagonLayer",
        df,
        get_hexagon="H3",
        get_fill_color="COLOR",
        get_line_color="COLOR",
        opacity=0.5,
        extruded=False
    )

st.title("Cell Towers by H3 (local CSV)")

col1, col2 = st.columns(2)

with col1:
    h3_resolution = st.slider("H3 resolution", min_value=1, max_value=6, value=3, step=1)

with col2:
    style_option = st.selectbox("Style schema", ("Contrast", "Snowflake"), index=1)

df = get_h3_df(h3_resolution)

# Build color scale
if style_option == "Contrast":
    # keep counts-based quantiles and ensure colors length == quantiles length
    quantiles = df["COUNT"].quantile([0, 0.25, 0.5, 0.75, 1])
    colors = ['gray', 'blue', 'green', 'orange', 'red']  # 5 colors to match 5 quantiles
else:  # "Snowflake"
    quantiles = df["COUNT"].quantile([0, 0.33, 0.66, 1])
    colors = ['#666666', '#24BFF2', '#126481', '#D966FF']  # 4 colors to match 4 quantiles

color_map = cm.LinearColormap(colors, vmin=float(quantiles.min()), vmax=float(quantiles.max()), index=quantiles.tolist())
df["COLOR"] = df["COUNT"].apply(lambda v: color_map.rgb_bytes_tuple(float(v)))

st.pydeck_chart(
    pdk.Deck(
        map_provider='carto',
        map_style='light',
        initial_view_state=pdk.ViewState(
            latitude=37.51405689475766,
            longitude=-96.50284957885742,
            zoom=3,
            height=430
        ),
        layers=[get_h3_layer(df)]
    )
)
