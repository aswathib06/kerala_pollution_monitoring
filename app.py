# app.py — Kerala Pollution Dashboard (Monthly & Seasonal Kriging + Yearly Animation)
# Revised for Streamlit Cloud (safe installs, caching, graceful failure handling)

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import logging
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from pykrige.ok import OrdinaryKriging
import plotly.express as px
import gdown
import requests
from requests.auth import HTTPBasicAuth
from typing import Optional, Tuple

# ---------- Streamlit page config ----------
st.set_page_config(
    page_title="Kerala Pollution Dashboard — Kriging & AI Assistant",
    layout="wide",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Config / constants ----------
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

PLANET_AUTH_URL = "https://api.planet.com/oauth/token"

# ---------- Utilities & caching ----------

@st.cache_data(show_spinner=False)
def load_kerala_polygon() -> Optional[MultiPolygon]:
    """
    Load Kerala polygon from local file or remote raw GitHub.
    Returns unary_union of polygons or None on failure.
    """
    try:
        if os.path.exists(BOUNDARY_PATH):
            with open(BOUNDARY_PATH, "r", encoding="utf-8") as f:
                gj = json.load(f)
        else:
            with requests.get(GITHUB_RAW_BOUNDARY, timeout=20) as resp:
                resp.raise_for_status()
                gj = resp.json()
    except Exception as e:
        logger.exception("Failed to load Kerala boundary geojson: %s", e)
        return None

    features = gj.get("features", [gj])
    polys = []
    for feat in features:
        try:
            geom = feat.get("geometry") if isinstance(feat, dict) else None
            if geom:
                shp = shape(geom)
                if isinstance(shp, (Polygon, MultiPolygon)):
                    polys.append(shp)
        except Exception:
            continue

    if not polys:
        return None

    return unary_union(polys)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load CSV from known local paths, fallback to gdown download, or create a small demo dataset.
    This avoids total failure during deployment when external download is blocked.
    """
    # 1) Look for files in LOCAL_DATA_PATHS
    for p in LOCAL_DATA_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                logger.info("Loaded local CSV: %s", p)
                return _prepare_df(df)
            except Exception:
                continue

    # 2) Local file in repo
    if os.path.exists(LOCAL_FILE):
        try:
            df = pd.read_csv(LOCAL_FILE)
            logger.info("Loaded repo CSV: %s", LOCAL_FILE)
            return _prepare_df(df)
        except Exception:
            pass

    # 3) Try to download using gdown
    try:
        with st.spinner("Downloading dataset..."):
            gdown.download(DATA_URL, LOCAL_FILE, quiet=True)
        if os.path.exists(LOCAL_FILE):
            df = pd.read_csv(LOCAL_FILE)
            logger.info("Downloaded and loaded CSV via gdown")
            return _prepare_df(df)
    except Exception as e:
        logger.exception("gdown download failed: %s", e)

    # 4) Fallback — create small demo dataset covering Kerala-ish lat/lon
    st.warning(
        "Could not find or download dataset. Loading a small demo dataset so the app can run. "
        "Replace with your real CSV in the repo or ensure the download link is accessible."
    )
    demo = _make_demo_dataframe()
    return _prepare_df(demo)


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    # sanitize column names and types
    df.columns = [c.strip() for c in df.columns]
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
    else:
        # create a date column if missing (demo fallback)
        df["date"] = pd.Timestamp("2019-01-01")

    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")

    # drop rows without coordinates or date
    df = df.dropna(subset=["date", "lat", "lon"])
    return df


def _make_demo_dataframe(n=500) -> pd.DataFrame:
    # Kerala approx lat: 8-12.6, lon: 74.3-77.6
    rng = np.random.default_rng(42)
    lats = rng.uniform(8.0, 12.5, size=n)
    lons = rng.uniform(74.5, 77.2, size=n)
    dates = pd.date_range("2018-01-01", periods=365).to_series().sample(n, replace=True, random_state=42).values
    df = pd.DataFrame({
        "date": dates,
        "lat": lats,
        "lon": lons,
        "NO2": rng.normal(30, 10, size=n).clip(1, None),
        "AOD": rng.normal(0.2, 0.05, size=n).clip(0, None)
    })
    return df


def clip_points_to_polygon(df: pd.DataFrame, polygon: MultiPolygon) -> pd.DataFrame:
    """
    Keep only points inside polygon. Vectorized-ish via list comprehension of shapely Points.
    For very large datasets, consider a spatial index or reducing sample first.
    """
    if polygon is None:
        return df  # nothing to clip
    pts = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    mask = np.array([polygon.contains(p) for p in pts])
    return df.loc[mask].reset_index(drop=True)


def detrend_linear(df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit simple linear plane (intercept + lon + lat) and return residuals + coefficients.
    """
    X = np.vstack([np.ones(len(df)), df["lon"].values, df["lat"].values]).T
    y = df[value_col].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    trend = X.dot(coef)
    return y - trend, coef


@st.cache_data(show_spinner=False)
def do_ordinary_kriging_on_residuals(
    df_points: pd.DataFrame,
    value_col: str,
    grid_res: int = DEFAULT_GRID,
    variogram_model: str = "spherical",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform Ordinary Kriging on residual values.
    Returns gx, gy, z, ss OR None if not enough points.
    """
    if len(df_points) < 3:
        return None

    lons = df_points["lon"].values
    lats = df_points["lat"].values
    vals = df_points[value_col].values

    pad_x = (lons.max() - lons.min()) * 0.02 if lons.max() != lons.min() else 0.01
    pad_y = (lats.max() - lats.min()) * 0.02 if lats.max() != lats.min() else 0.01

    gx = np.linspace(lons.min() - pad_x, lons.max() + pad_x, grid_res)
    gy = np.linspace(lats.min() - pad_y, lats.max() + pad_y, grid_res)

    try:
        OK = OrdinaryKriging(lons, lats, vals, variogram_model=variogram_model, verbose=False, enable_plotting=False)
        z, ss = OK.execute("grid", gx, gy)
        return gx, gy, z, ss
    except Exception as e:
        logger.exception("Kriging failed: %s", e)
        return None


def mask_grid_to_polygon(gx: np.ndarray, gy: np.ndarray, z: np.ndarray, polygon: MultiPolygon) -> pd.DataFrame:
    """
    Mask grid cells that fall outside polygon and return DataFrame of lon, lat, value.
    """
    xx, yy = np.meshgrid(gx, gy)
    lon = xx.ravel()
    lat = yy.ravel()
    val = z.ravel()
    pts = [Point(xy) for xy in zip(lon, lat)]
    mask = np.array([polygon.contains(p) for p in pts])
    return pd.DataFrame({"lon": lon[mask], "lat": lat[mask], "value": val[mask]})


# ---------- Planet API helpers ----------
def planet_authenticate(client_id: str, client_secret: str) -> Optional[str]:
    """Get OAuth2 token from Planet using client credentials."""
    try:
        resp = requests.post(
            PLANET_AUTH_URL,
            data={"grant_type": "client_credentials"},
            auth=HTTPBasicAuth(client_id, client_secret),
            timeout=20
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
        else:
            logger.error("Planet auth failed: %s %s", resp.status_code, resp.text)
            return None
    except Exception as e:
        logger.exception("Planet auth exception: %s", e)
        return None


def planet_search_by_date(token: str, aoi: dict, start_date: str, end_date: str, max_cloud: int = 20):
    """Quick-search Planet API for PSScene within AOI and date range."""
    url = "https://api.planet.com/data/v1/quick-search"
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "item_types": ["PSScene"],
        "filter": {
            "type": "AndFilter",
            "config": [
                {
                    "type": "GeometryFilter",
                    "field_name": "geometry",
                    "config": aoi
                },
                {
                    "type": "DateRangeFilter",
                    "field_name": "acquired",
                    "config": {
                        "gte": f"{start_date}T00:00:00Z",
                        "lte": f"{end_date}T23:59:59Z"
                    }
                },
                {
                    "type": "RangeFilter",
                    "field_name": "cloud_cover",
                    "config": {"lte": max_cloud}
                }
            ]
        }
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": resp.text, "status_code": resp.status_code}
    except Exception as e:
        logger.exception("Planet search failed: %s", e)
        return {"error": str(e)}


# ---------- Load data & polygon ----------
df_all = load_data()
kerala_poly = load_kerala_polygon()

if kerala_poly is None:
    st.error("Kerala boundary not available — polygon could not be loaded. "
             "Please add kerala_boundary.geojson to the repo or check the raw GitHub URL.")
    st.stop()

# ---------- UI Controls ----------
st.sidebar.header("Controls")
candidate = [c for c in df_all.columns if c.upper() in ["AOD", "NO2", "SO2", "CO", "O3"]]
if not candidate:
    # fallback: choose any numeric column
    candidate = [c for c in df_all.select_dtypes(include=[np.number]).columns.tolist() if c not in ["lat", "lon"]]
if not candidate:
    st.error("No numeric pollutant-like columns found in dataset. Add columns like 'NO2' or 'AOD'.")
    st.stop()

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
variogram_model = st.sidebar.selectbox("Variogram model", ["spherical", "exponential", "gaussian"])
use_log = st.sidebar.checkbox("Log-transform pollutant", value=False)

# Planet API login (prefers st.secrets)
st.sidebar.subheader("🌍 Planet API Login")
_secrets_client = st.secrets.get("PLANET_CLIENT_ID") if st.secrets else None
_secrets_secret = st.secrets.get("PLANET_CLIENT_SECRET") if st.secrets else None

client_id_input = st.sidebar.text_input("Planet Client ID (only if not using secrets)", type="password")
client_secret_input = st.sidebar.text_input("Planet Client Secret (only if not using secrets)", type="password")
use_secrets = bool(_secrets_client and _secrets_secret)

if use_secrets:
    client_id = _secrets_client
    client_secret = _secrets_secret
else:
    client_id = client_id_input
    client_secret = client_secret_input

if st.sidebar.button("Login to Planet API"):
    if not client_id or not client_secret:
        st.sidebar.error("Provide client id & secret (or set them in Streamlit secrets).")
    else:
        token = planet_authenticate(client_id, client_secret)
        if token:
            st.session_state["planet_token"] = token
            st.sidebar.success("Planet login OK — token stored in session")
        else:
            st.sidebar.error("Planet login failed — check credentials")

# ---------- Simple Gen-AI style assistant ----------
st.sidebar.markdown("### 🧠 Ask a Pollution Question")
user_question = st.sidebar.text_input("Ask any question (e.g., 'Which place has highest NO2?')")


def answer_pollution_question(question: str, df: pd.DataFrame) -> str:
    q = question.lower().strip()
    pol = pollutant
    if q == "":
        return ""

    try:
        if "highest" in q or "hotspot" in q or "high" in q:
            row = df.loc[df[pol].idxmax()]
            return f"🔥 Highest {pol} detected near: lat {row['lat']:.3f}, lon {row['lon']:.3f}, value {row[pol]:.2f}"
        if "lowest" in q or "cleanest" in q:
            row = df.loc[df[pol].idxmin()]
            return f"🌿 Lowest {pol} detected near: lat {row['lat']:.3f}, lon {row['lon']:.3f}, value {row[pol]:.2f}"
        if "average" in q or "mean" in q:
            return f"📊 Average {pol}: {df[pol].mean():.2f}"
        if "trend" in q:
            df2 = df.copy()
            df2["year"] = df2["date"].dt.year
            trend = df2.groupby("year")[pol].mean().sort_index()
            if len(trend) < 2:
                return "Not enough years to compute a trend."
            return "📈 Increasing trend" if trend.iloc[-1] > trend.iloc[0] else "📉 Decreasing trend"
    except Exception as e:
        logger.exception("Assistant failed: %s", e)
        return "Sorry — couldn't answer that question."

    return "Try keywords: highest, lowest, average, trend."


# ---------- Filtering / slicing ----------
if view_mode == "Monthly Mean Kriging":
    df_all["year_month"] = df_all["date"].dt.to_period("M").astype(str)
    sel = st.sidebar.selectbox("Month", sorted(df_all["year_month"].unique()))
    df_slice = df_all[df_all["year_month"] == sel]

elif view_mode == "Seasonal Kriging":
    def get_season(d):
        m = d.month
        if m in [12, 1, 2]:
            return "Winter"
        if m in [3, 4, 5]:
            return "Summer"
        if m in [6, 7, 8, 9]:
            return "Monsoon"
        return "Post-monsoon"
    df_all["season"] = df_all["date"].apply(get_season)
    sel = st.sidebar.selectbox("Season", ["Winter", "Summer", "Monsoon", "Post-monsoon"])
    df_slice = df_all[df_all["season"] == sel]

elif view_mode == "Daily Slice (points only)":
    dmin, dmax = df_all["date"].min().date(), df_all["date"].max().date()
    sel_date = st.sidebar.date_input("Select date", value=dmin, min_value=dmin, max_value=dmax)
    df_slice = df_all[df_all["date"].dt.date == sel_date]

else:
    df_slice = df_all.copy()

df_slice = df_slice.dropna(subset=["lat", "lon", pollutant])
df_slice = clip_points_to_polygon(df_slice, kerala_poly)

# sample points for plotting speed
df_sample = df_slice.sample(min(sample_size, max(1, len(df_slice))), random_state=42) if len(df_slice) else df_slice

# ---------- Title & assistant ----------
st.title("Kerala Pollution Dashboard — Kriging + AI Assistant")
if user_question:
    st.subheader("🧠 Gen-AI Pollution Assistant")
    st.info(answer_pollution_question(user_question, df_slice))

# ---------- Planet search UI ----------
st.subheader("🌍 Search Planet Imagery (Kerala)")

if "planet_token" not in st.session_state:
    st.info("Login to Planet API using the sidebar.")
else:
    kerala_aoi = {
        "type": "Polygon",
        "coordinates": [[
            [74.5, 7.8],
            [77.5, 7.8],
            [77.5, 12.9],
            [74.5, 12.9],
            [74.5, 7.8]
        ]]
    }

    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")
    if st.button("Search Kerala Imagery"):
        if start_date > end_date:
            st.error("Start date must be before or equal to end date.")
        else:
            with st.spinner("Searching satellite images..."):
                results = planet_search_by_date(
                    st.session_state.get("planet_token"),
                    kerala_aoi,
                    start_date.isoformat(),
                    end_date.isoformat()
                )

            if results is None:
                st.error("Search failed (no response).")
            elif "error" in results:
                st.error(f"Search error (http {results.get('status_code')}): {results.get('error')}")
            elif "features" in results:
                st.success(f"Found {len(results['features'])} images")
                for feat in results["features"][:20]:
                    props = feat.get("properties", {})
                    st.write("### Image ID:", feat.get("id"))
                    st.write("Acquired:", props.get("acquired"))
                    st.write("Cloud cover:", props.get("cloud_cover"))
                    st.markdown("---")
            else:
                st.warning("No images found for this date range.")

# ---------- Visual modes ----------
if view_mode == "Interactive Map":
    if df_sample.empty:
        st.warning("No points available to display.")
    else:
        fig = px.scatter_mapbox(df_sample, lat="lat", lon="lon", color=pollutant,
                                zoom=7, height=700, color_continuous_scale="Turbo")
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Heatmap":
    if df_sample.empty:
        st.warning("No points available to display.")
    else:
        fig = px.density_mapbox(df_sample, lat="lat", lon="lon", z=pollutant, radius=20,
                                zoom=7, height=700, color_continuous_scale="Turbo")
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

elif view_mode in ["Monthly Mean Kriging", "Seasonal Kriging"]:
    if df_sample.empty or len(df_sample) < 3:
        st.warning("Not enough points to perform kriging. Need at least 3 points. Try increasing sample size or change date/season.")
    else:
        # detrend
        df_pts = df_sample.copy().reset_index(drop=True)
        df_pts["val"] = df_pts[pollutant].astype(float)
        resid, coef = detrend_linear(df_pts, "val")
        df_pts["resid"] = resid

        krig_result = do_ordinary_kriging_on_residuals(df_pts, "resid", grid_res=grid_res, variogram_model=variogram_model)
        if krig_result is None:
            st.error("Kriging failed or insufficient points. Try a coarser grid or larger sample.")
        else:
            gx, gy, z_resid, ss = krig_result
            trend_grid = predict_trend_grid = None
            try:
                # recreate trend grid from coef
                GX, GY = np.meshgrid(gx, gy)
                X = np.vstack([np.ones(GX.size), GX.ravel(), GY.ravel()]).T
                trend_grid = X.dot(coef).reshape(GX.shape)
            except Exception as e:
                logger.exception("Could not compute trend grid: %s", e)
                trend_grid = np.zeros_like(z_resid)

            z_total = z_resid + trend_grid
            grid_df = mask_grid_to_polygon(gx, gy, z_total, kerala_poly)
            if grid_df.empty:
                st.warning("Kriging produced no values inside Kerala polygon.")
            else:
                fig = px.density_mapbox(grid_df, lat="lat", lon="lon", z="value", radius=8,
                                        zoom=7, height=700, color_continuous_scale="Turbo")
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Yearly Heatmap Animation (2018–2025)":
    df_year = df_all.copy()
    df_year["year"] = df_year["date"].dt.year
    df_year = df_year[df_year["year"].between(2018, 2025)]

    if df_year.empty:
        st.warning("No yearly data available for animation.")
    else:
        max_year = st.sidebar.slider("Max points per year", 2000, 10000, 4000)
        df_anim = df_year.groupby("year").apply(lambda g: g.sample(min(max_year, len(g)))).reset_index(drop=True)

        fig = px.density_mapbox(df_anim, lat="lat", lon="lon", z=pollutant,
                                animation_frame="year", radius=18, zoom=7,
                                height=750, color_continuous_scale="Turbo")
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

# ---------- End ----------
st.markdown("---")
st.caption("If you run into errors during deployment, check Streamlit logs for the pip install step. "
           "Make sure runtime.txt (python-3.11.4) and the pinned requirements.txt are present in the repo root.")
