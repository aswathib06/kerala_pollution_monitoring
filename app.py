# app.py — Kerala Pollution Dashboard (Monthly & Seasonal Kriging + Yearly Animation)
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from pykrige.ok import OrdinaryKriging
import plotly.express as px
import gdown

st.set_page_config(page_title="Kerala Pollution Dashboard — Kriging & AI Assistant", layout="wide")

# -------------------------
# CONFIG
# -------------------------
LOCAL_DATA_PATHS = [
    "/mnt/data/df_final.csv",
    "/mnt/data/Ernakulam_Daily_AQI_2018_2024_with_LatLon.csv",
    "/mnt/data/Kerala_S5P_Cleaned_2018_2025.csv"
]
DATA_URL = "https://drive.google.com/uc?id=1M6I2ku_aWGkWz0GypktKXeRJPjNhlsM2"  
LOCAL_FILE = "kerala_pollution.csv"

BOUNDARY_PATH = "kerala_boundary.geojson"
GITHUB_RAW_BOUNDARY = "https://raw.githubusercontent.com/Abhinand-1/air_pollution/main/kerala_boundary.geojson"

DEFAULT_SAMPLE = 1000
DEFAULT_GRID = 60

# -------------------------
# DATA LOADING
# -------------------------
@st.cache_data(show_spinner=False)
def find_local_csv():
    for p in LOCAL_DATA_PATHS:
        if os.path.exists(p):
            return p
    return None

@st.cache_data(show_spinner=False)
def load_data():
    local = find_local_csv()
    if local:
        csv_path = local
    else:
        if os.path.exists(LOCAL_FILE):
            csv_path = LOCAL_FILE
        else:
            with st.spinner("Downloading dataset..."):
                gdown.download(DATA_URL, LOCAL_FILE, quiet=False)
                csv_path = LOCAL_FILE

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="ignore")

    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")

    df = df.dropna(subset=["date","lat","lon"])
    return df

@st.cache_data(show_spinner=False)
def load_kerala_polygon():
    try:
        if os.path.exists(BOUNDARY_PATH):
            with open(BOUNDARY_PATH, "r", encoding="utf-8") as f:
                gj = json.load(f)
        else:
            import urllib.request
            with urllib.request.urlopen(GITHUB_RAW_BOUNDARY) as resp:
                gj = json.load(resp)
    except:
        st.error("Boundary geojson missing.")
        st.stop()

    features = gj["features"] if "features" in gj else [gj]
    polys = []

    for feat in features:
        shp = shape(feat["geometry"])
        if isinstance(shp, (Polygon, MultiPolygon)):
            polys.append(shp)

    return unary_union(polys)

def clip_points_to_polygon(df, polygon):
    pts = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    mask = np.array([polygon.contains(p) for p in pts])
    return df.loc[mask]

def detrend_linear(df, value_col):
    X = np.vstack([np.ones(len(df)), df["lon"], df["lat"]]).T
    y = df[value_col].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    trend = X.dot(coef)
    return y - trend, coef

def predict_trend_grid(gx, gy, coef):
    GX, GY = np.meshgrid(gx, gy)
    X = np.vstack([np.ones(GX.size), GX.ravel(), GY.ravel()]).T
    trend = X.dot(coef).reshape(GX.shape)
    return trend

def do_ordinary_kriging_on_residuals(df_points, value_col, grid_res=DEFAULT_GRID, variogram_model="spherical"):
    lons = df_points["lon"].values
    lats = df_points["lat"].values
    vals = df_points[value_col].values

    pad_x = (lons.max() - lons.min()) * 0.02
    pad_y = (lats.max() - lats.min()) * 0.02

    gx = np.linspace(lons.min() - pad_x, lons.max() + pad_x, grid_res)
    gy = np.linspace(lats.min() - pad_y, lats.max() + pad_y, grid_res)

    OK = OrdinaryKriging(lons, lats, vals, variogram_model=variogram_model, verbose=False, enable_plotting=False)
    z, ss = OK.execute("grid", gx, gy)
    return gx, gy, z, ss

def mask_grid_to_polygon(gx, gy, z, polygon):
    xx, yy = np.meshgrid(gx, gy)
    lon = xx.ravel()
    lat = yy.ravel()
    val = z.ravel()
    pts = [Point(xy) for xy in zip(lon, lat)]
    mask = np.array([polygon.contains(p) for p in pts])
    return pd.DataFrame({"lon": lon[mask], "lat": lat[mask], "value": val[mask]})

# -------------------------
# LOAD EVERYTHING
# -------------------------
df_all = load_data()
kerala_poly = load_kerala_polygon()

# -------------------------
# UI CONTROLS
# -------------------------
st.sidebar.header("Controls")

candidate = [c for c in df_all.columns if c.upper() in ["AOD","NO2","SO2","CO","O3"]]
pollutant = st.sidebar.selectbox("Pollutant", candidate)

view_mode = st.sidebar.radio("View Mode", [
    "Interactive Map",
    "Monthly Mean Kriging",
    "Seasonal Kriging",
    "Heatmap",
    "Yearly Heatmap Animation (2018–2025)",
    "Daily Slice (points only)"
])

sample_size = st.sidebar.slider("Sample size", 200, 2000, DEFAULT_SAMPLE)
grid_res = st.sidebar.slider("Grid resolution", 40, 120, DEFAULT_GRID)
variogram_model = st.sidebar.selectbox("Variogram model", ["spherical","exponential","gaussian"])
use_log = st.sidebar.checkbox("Log-transform pollutant", value=False)

# ---------------------------------------------------
# 🧠 GEN-AI STYLE QUESTION ANSWERING MODULE
# ---------------------------------------------------
st.sidebar.markdown("### 🧠 Ask a Pollution Question")
user_question = st.sidebar.text_input("Ask any question (e.g., 'Which place has highest NO2?')")

def answer_pollution_question(question, df):
    question_lower = question.lower()
    pol = pollutant

    if "highest" in question_lower or "hotspot" in question_lower or "high" in question_lower:
        row = df.loc[df[pol].idxmax()]
        return (
            f"🔥 **Highest {pol} level** detected near:\n\n"
            f"• Latitude: {row['lat']:.3f}\n"
            f"• Longitude: {row['lon']:.3f}\n"
            f"• Value: {row[pol]:.2f}\n"
        )

    if "lowest" in question_lower or "cleanest" in question_lower:
        row = df.loc[df[pol].idxmin()]
        return (
            f"🌿 **Lowest {pol} level** detected near:\n\n"
            f"• Latitude: {row['lat']:.3f}\n"
            f"• Longitude: {row['lon']:.3f}\n"
            f"• Value: {row[pol]:.2f}\n"
        )

    if "average" in question_lower or "mean" in question_lower:
        return f"📊 Average {pol}: {df[pol].mean():.2f}"

    if "trend" in question_lower:
        df2 = df.copy()
        df2["year"] = df2["date"].dt.year
        trend = df2.groupby("year")[pol].mean()
        return "📈 Increasing trend" if trend.iloc[-1] > trend.iloc[0] else "📉 Decreasing trend"

    return "❓ Try asking: highest, lowest, average, or trend."

# -------------------------
# FILTERING
# -------------------------
if view_mode == "Monthly Mean Kriging":
    df_all["year_month"] = df_all["date"].dt.to_period("M").astype(str)
    sel = st.sidebar.selectbox("Month", sorted(df_all["year_month"].unique()))
    df_slice = df_all[df_all["year_month"] == sel]

elif view_mode == "Seasonal Kriging":
    def get_season(d):
        m = d.month
        return "Winter" if m in [12,1,2] else "Summer" if m in [3,4,5] else "Monsoon" if m in [6,7,8,9] else "Post-monsoon"
    df_all["season"] = df_all["date"].apply(get_season)
    sel = st.sidebar.selectbox("Season", ["Winter","Summer","Monsoon","Post-monsoon"])
    df_slice = df_all[df_all["season"] == sel]

elif view_mode == "Daily Slice (points only)":
    dmin, dmax = df_all["date"].min().date(), df_all["date"].max().date()
    sel_date = st.sidebar.date_input("Select date", value=dmin, min_value=dmin, max_value=dmax)
    df_slice = df_all[df_all["date"].dt.date == sel_date]

else:
    df_slice = df_all

df_slice = df_slice.dropna(subset=["lat","lon",pollutant])
df_slice = clip_points_to_polygon(df_slice, kerala_poly)

df_sample = df_slice.sample(min(sample_size, len(df_slice)), random_state=42)

# -------------------------
# TITLE
# -------------------------
st.title("Kerala Pollution Dashboard — Kriging + AI Assistant")

# AI Answer Display
if user_question:
    st.subheader("🧠 Gen-AI Pollution Assistant")
    st.info(answer_pollution_question(user_question, df_slice))

# -------------------------
# VISUAL MODES
# -------------------------
if view_mode == "Interactive Map":
    fig = px.scatter_mapbox(df_sample, lat="lat", lon="lon", color=pollutant,
                            zoom=7, height=700, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Heatmap":
    fig = px.density_mapbox(df_sample, lat="lat", lon="lon", z=pollutant, radius=20,
                            zoom=7, height=700, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

elif view_mode in ["Monthly Mean Kriging","Seasonal Kriging"]:
    df_pts = df_sample.copy()
    df_pts["val"] = df_pts[pollutant]
    resid, coef = detrend_linear(df_pts, "val")
    df_pts["resid"] = resid

    gx, gy, z_resid, ss = do_ordinary_kriging_on_residuals(df_pts, "resid", grid_res=grid_res, variogram_model=variogram_model)
    trend_grid = predict_trend_grid(gx, gy, coef)

    z_total = z_resid + trend_grid
    grid_df = mask_grid_to_polygon(gx, gy, z_total, kerala_poly)

    fig = px.density_mapbox(grid_df, lat="lat", lon="lon", z="value", radius=8,
                            zoom=7, height=700, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Yearly Heatmap Animation (2018–2025)":
    df_year = df_all.copy()
    df_year["year"] = df_year["date"].dt.year
    df_year = df_year[df_year["year"].between(2018,2025)]

    max_year = st.sidebar.slider("Max points per year", 2000, 10000, 4000)
    df_anim = df_year.groupby("year").apply(lambda g: g.sample(min(max_year, len(g)))).reset_index(drop=True)

    fig = px.density_mapbox(df_anim, lat="lat", lon="lon", z=pollutant,
                            animation_frame="year", radius=18, zoom=7,
                            height=750, color_continuous_scale="Turbo")
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
