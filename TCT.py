import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xarray as xr
import rasterio
import tempfile
import os
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px

# Streamlit page configuration
st.set_page_config(page_title="TERI Climate Tool Demo", page_icon="🌏", layout="wide")
st.logo("data/TERI Logo Seal.png", size="large")

# ------------------------------------
# Section A: Configuration
# ------------------------------------
DEFAULT_FILES = {
    "Rainfall 1994-2023": {
        "avg": "data/IMD_RF_AVG_after1994.nc",
        "ts": "data/RF_yearly.nc",
        "no_data": -999
    },
    #"Rainfall 1994-2023 TIFF": {
    #    "avg": "data/IMD_RF_avg_1994-2023.tif",
    #    "ts": None,
    #    "no_data": -999
    #},
    #"Rainfall 1994-2023 CSV": {
    #    "avg": "data/IMD_RF_AVG_after1994.csv",
    #    "ts": None,
    #    "no_data": -999
    #},
    "Tmin Projection": {
        "avg": "data/Tmin-2081-2100_20yravg.nc",
        "ts": None,
        "no_data": 99.99
    },
    "Tmax Projection": {
        "avg": "data/Tmax-2081-2100_20yravg.nc",
        "ts": None,
        "no_data": 99.99
    },    

}
DEFAULT_SHAPEFILE = "data/India_State_Boundary.shp"

# ------------------------------------
# Section B: Utility Functions
# ------------------------------------
def load_data(file_config, file_option, no_data_value):
    """Loads data from a file and returns a DataFrame, file type, and metadata."""
    if file_option == "Upload Your Own Data":
        uploaded_file = st.sidebar.file_uploader("Upload Data", ["csv", "nc", "netcdf", "tif", "tiff"])
        if not uploaded_file:
            return None, None, None, None, []
        ext = uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name
    else:
        file_path = file_config["avg"]
        ext = file_path.split(".")[-1].lower()

    if ext == "csv":
        df = pd.read_csv(file_path).dropna()
        if no_data_value is not None:
            for col in df.select_dtypes(include=["float64", "int64"]).columns:
                df[col] = df[col].where(df[col] != no_data_value, np.nan)
        df = df.dropna()
        return df, "csv", file_path, uploaded_file.name if file_option == "Upload Your Own Data" else file_option, [col for col in df.columns if col not in ["lon", "lat"]]
    
    elif ext in ["nc", "netcdf"]:
        ds = xr.open_dataset(file_path)
        available_vars = list(ds.data_vars)
        # Ensure we only process float-based data variables
        float_vars = [var for var in available_vars if ds[var].dtype in ["float32", "float64"]]
        if not float_vars:
            st.error("No float-type variables found in the NetCDF file.")
            return None, None, None, None, []
        
        # Use the first float variable as default or allow user to select later
        selected_var = float_vars[0]
        if no_data_value is not None:
            # Apply no-data replacement only to the selected float variable
            ds[selected_var] = ds[selected_var].where(ds[selected_var] != no_data_value, np.nan)
        
        # Convert to DataFrame, ensuring lat/lon are included
        df = ds[selected_var].to_dataframe().reset_index().dropna()
        df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
        if "lat" not in df.columns or "lon" not in df.columns:
            st.error("NetCDF file must contain 'lat' and 'lon' coordinates.")
            return None, None, None, None, []
        
        return df, "netcdf", file_path, file_option, float_vars
    
    elif ext in ["tif", "tiff"]:
        with rasterio.open(file_path) as src:
            array = src.read(1)
            nodata = no_data_value or src.nodata
            array = np.where(array == nodata, np.nan, array)
            bounds, res = src.bounds, src.res
            lons = np.arange(bounds.left + res[0]/2, bounds.right, res[0])
            lats = np.arange(bounds.top - res[1]/2, bounds.bottom, -res[1])
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            df = pd.DataFrame({"lon": lon_grid.flatten(), "lat": lat_grid.flatten(), "value": array.flatten()}).dropna()
        return df, "tiff", file_path, file_option, ["value"]
    
    return None, None, None, None, []

def load_shapefile(shp_source, default_shapefile):
    """Loads a shapefile and returns a GeoDataFrame."""
    if shp_source == "Use Default Shapefile":
        return gpd.read_file(default_shapefile) if os.path.exists(default_shapefile) else None
    
    uploaded_shp = st.file_uploader("Upload Shapefile", ["shp"])
    if not uploaded_shp:
        return None
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        shp_path = os.path.join(tmp_dir, uploaded_shp.name)
        with open(shp_path, "wb") as f:
            f.write(uploaded_shp.read())
        for ext in [".shx", ".dbf", ".prj", ".cpg"]:
            if uploaded_assoc := st.session_state.get(f"shp_{ext}"):
                with open(os.path.join(tmp_dir, f"{os.path.splitext(uploaded_shp.name)[0]}{ext}"), "wb") as f:
                    f.write(uploaded_assoc.read())
        return gpd.read_file(shp_path)

def filter_by_region(df, gdf, region, region_col):
    """Filters DataFrame to points within the selected region's geometry."""
    if gdf is None or gdf.empty or not region:
        return df
    try:
        geometry = gdf[gdf[region_col] == region].geometry.iloc[0]
        points = gpd.GeoDataFrame(df, geometry=[Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])], crs=gdf.crs)
        return points[points.geometry.within(geometry)].drop(columns="geometry").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error filtering by region: {e}")
        return df
    
def get_region_bounds(gdf, region, region_col):
    """Returns the bounds of the selected region."""
    if gdf is None or gdf.empty or not region or region == "Select a region":
        return None
    try:
        geometry = gdf[gdf[region_col] == region].geometry.iloc[0]
        minx, miny, maxx, maxy = geometry.bounds
        return {"min_lon": minx, "max_lon": maxx, "min_lat": miny, "max_lat": maxy}
    except Exception as e:
        st.error(f"Error calculating bounds: {e}")
        return None

def calculate_zoom(lat_range, lon_range):
    """Sets map zoom based on data size."""
    max_range = max(lat_range, lon_range)
    
    if max_range > 100:  # Global view
        return 1
    elif max_range > 30:  # Continent view
        return 2
    elif max_range > 15:  # Country view (e.g., India)
        return 4
    elif max_range > 5:   # State view
        return 5
    elif max_range > 1:   # City view
        return 7
    elif max_range > 0.1: # Small area
        return 10
    else:                 # Very small area
        return 13
    
def read_time_series(file_path, lat, lon, color="blue"):
    """Reads NetCDF time series data and creates a plot."""
    try:
        ds = xr.open_dataset(file_path)
        var = list(ds.data_vars)[0]
        lat_idx = np.abs((ds.get("lat", ds.get("latitude")).values - lat)).argmin()
        lon_idx = np.abs((ds.get("lon", ds.get("longitude")).values - lon)).argmin()
        ts_data = ds[var].isel(lat=lat_idx, lon=lon_idx).to_dataframe().reset_index()
        fig = px.line(ts_data, x="year", y=var, title=f"Time Series at Lat: {lat:.2f}, Lon: {lon:.2f}",
                      labels={"year": "Time", var: "Value"}, color_discrete_sequence=[color])
        fig.update_layout(xaxis_title="Time", yaxis_title=var, height=600, width=1000)
        return fig
    except Exception as e:
        st.error(f"Error processing time series: {e}")
        return None

# ------------------------------------
# Section C: Visualization Functions
# ------------------------------------
def plot_heatmap(df, file_name, vis_config, value_col):
    """Creates a heatmap using Plotly Choroplethmapbox."""
    center = {"lat": df["lat"].mean(), "lon": df["lon"].mean()}
    zoom = calculate_zoom(df["lat"].max() - df["lat"].min(), df["lon"].max() - df["lon"].min())
    
    features = [
        {
            "type": "Feature", "id": i,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [row["lon"] - vis_config["cell_width"]/2, row["lat"] - vis_config["cell_height"]/2],
                    [row["lon"] + vis_config["cell_width"]/2, row["lat"] - vis_config["cell_height"]/2],
                    [row["lon"] + vis_config["cell_width"]/2, row["lat"] + vis_config["cell_height"]/2],
                    [row["lon"] - vis_config["cell_width"]/2, row["lat"] + vis_config["cell_height"]/2],
                    [row["lon"] - vis_config["cell_width"]/2, row["lat"] - vis_config["cell_height"]/2]
                ]]
            },
            "properties": {"value": row[value_col], "lat": row["lat"], "lon": row["lon"]}
        }
        for i, row in df.iterrows()
    ]
    
    geojson = {"type": "FeatureCollection", "features": features}
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson, locations=[f["id"] for f in features], z=[f["properties"]["value"] for f in features],
        colorscale=vis_config["color_scale"], marker_opacity=vis_config["grid_opacity"], marker_line_width=0,
        colorbar=dict(title=value_col),
        customdata=np.stack(([f["properties"]["lat"] for f in features], [f["properties"]["lon"] for f in features]), axis=-1),
        hovertemplate="<b>Lat: %{customdata[0]:.5f}</b><br>Lon: %{customdata[1]:.5f}<br>Value: %{z}<extra></extra>"
    ))
    fig.update_layout(mapbox=dict(style=vis_config["map_style"], center=center, zoom=zoom), height=900, width=1000, title=f"Heatmap - {file_name}")
    return fig

def plot_3d(df, file_type, file_path, file_name, vis_config, value_col, region_config=None):
    """Plots a 3D surface or scatter plot based on file type."""
    if file_type == "csv":
        is_grid = st.checkbox("Is CSV data on a regular grid?", value=False)
        x, y, z = df["lon"], df["lat"], df["value"]  # Use 'value' for CSV since DataFrame is renamed
        fig = go.Figure(
            go.Surface(z=z.values.reshape(len(df["lat"].unique()), -1), x=x.unique(), y=y.unique(), colorscale=vis_config["color_scale"])
            if is_grid else
            go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=3, color=z, colorscale=vis_config["color_scale"], opacity=0.8))
        )
    elif file_type == "netcdf":
        ds = xr.open_dataset(file_path)
        lons, lats, data = ds["lon"].values, ds["lat"].values, ds[value_col].squeeze().values  # Use original value_col
        if region_config and region_config["region"]:
            bounds = get_region_bounds(region_config["gdf"], region_config["region"], region_config["region_col"])
            if bounds:
                lon_mask = (lons >= bounds["min_lon"]) & (lons <= bounds["max_lon"])
                lat_mask = (lats >= bounds["min_lat"]) & (lats <= bounds["max_lat"])
                lons, lats, data = lons[lon_mask], lats[lat_mask], data[np.ix_(lat_mask, lon_mask)]
        fig = go.Figure(go.Surface(z=data, x=lons, y=lats, colorscale=vis_config["color_scale"]))
    elif file_type == "tiff":
        with rasterio.open(file_path) as src:
            array, bounds, res = src.read(1), src.bounds, src.res
            array = np.where(array == src.nodata, np.nan, array)
            lons = np.arange(bounds.left + res[0]/2, bounds.right, res[0])
            lats = np.arange(bounds.top - res[1]/2, bounds.bottom, -res[1])
            if region_config and region_config["region"]:
                bounds = get_region_bounds(region_config["gdf"], region_config["region"], region_config["region_col"])
                if bounds:
                    lon_mask = (lons >= bounds["min_lon"]) & (lons <= bounds["max_lon"])
                    lat_mask = (lats >= bounds["min_lat"]) & (lats <= bounds["max_lat"])
                    lons, lats, array = lons[lon_mask], lats[lat_mask], array[np.ix_(lat_mask, lon_mask)]
        fig = go.Figure(go.Surface(z=array, x=lons, y=lats, colorscale=vis_config["color_scale"]))
    
    fig.update_layout(title=f"3D {'Surface' if file_type != 'csv' or is_grid else 'Scatter'} - {file_name}",
                      scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title=value_col), height=900)
    return fig


def plot_scatter(df, file_name, vis_config, value_col):
    """Plots a scatter map for CSV data."""
    zoom = calculate_zoom(df["lat"].max() - df["lat"].min(), df["lon"].max() - df["lon"].min())
    fig = go.Figure(go.Scattermapbox(
        lat=df["lat"], lon=df["lon"], mode="markers",
        marker=dict(size=8, color=df[value_col], colorscale=vis_config["color_scale"], opacity=0.7, colorbar=dict(title=value_col)),
        hovertemplate="<b>Lat: %{lat:.5f}</b><br>Lon: %{lon:.5f}<br>Value: %{marker.color}<extra></extra>"
    ))
    fig.update_layout(mapbox=dict(style=vis_config["map_style"], center=dict(lat=df["lat"].mean(), lon=df["lon"].mean()),
                                  zoom=zoom),
                      height=900, width=1000, title=f"Scatter Plot - {file_name}")
    return fig

# ------------------------------------
# Section D: UI and Settings
# ------------------------------------
def setup_ui():
    """Sets up the Streamlit UI and sidebar controls."""
    st.markdown("<h1 style='color: #dc6142'>TERI Climate Tool: Demo</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='color: #dc6142'>Controls</h2>", unsafe_allow_html=True)
    
    file_option = st.sidebar.selectbox("Select File", list(DEFAULT_FILES.keys()) + ["Upload Your Own Data"])
    
    data_config = {"show_raw": False}
    vis_config = {
        "cell_width": 0.25,
        "cell_height": 0.25,
        "grid_opacity": 0.8,
        "map_style": "carto-positron",
        "color_scale": "Blues"
    }
    
    return file_option, data_config, vis_config

# ... (other imports and code remain unchanged)

def setup_data_settings(columns, file_option):
    """Sets up data settings including value column and no-data value."""
    with st.sidebar.expander("Data Settings"):
        value_col = st.selectbox("Select Variable", columns) if columns else None
        default_no_data = DEFAULT_FILES.get(file_option, {}).get("no_data", -9999)
        step = 0.01 if isinstance(default_no_data, float) else 1
        no_data_value = st.number_input(
            "No-Data Value",
            value=default_no_data,
            step=step,
            format="%f" if isinstance(default_no_data, float) else "%d"
        )
        data_config = {"show_raw": st.checkbox("Show Raw Data")}
    return value_col, no_data_value, data_config


def setup_visualization_settings():
    """Sets up visualization settings."""
    with st.sidebar.expander("Advanced Visualization Settings"):
        vis_config = {
            "cell_width": st.number_input("Cell Width", value=0.25, step=0.001),
            "cell_height": st.number_input("Cell Height", value=0.25, step=0.001),
            "grid_opacity": st.number_input("Grid Opacity", 0.0, 1.0, 0.8, 0.05),
            "map_style": st.selectbox("Map Style", ["carto-darkmatter", "carto-positron","open-street-map", "stamen-terrain"], help="carto-darkmatter provides a dark mode map style."),
            "color_scale": st.selectbox("Color Scale", ["Blues", "Viridis", "Plasma", "Inferno", "Cividis", "Reds", "Hot", "Electric"])
        }
    return vis_config
def setup_time_series(file_option):
    """Sets up time series settings."""
    if file_option in DEFAULT_FILES and DEFAULT_FILES[file_option].get("ts"):
        with st.sidebar.expander("Time Series Settings"):
            lat = st.number_input("Latitude", -90.0, 90.0, 28.5, 0.5)
            lon = st.number_input("Longitude", -180.0, 180.0, 77.5, 0.5)
            color = st.selectbox("Line Color", ["blue", "red", "green", "purple", "orange", "black", "cyan", "magenta"])
            if st.button("Generate Time Series"):
                return read_time_series(DEFAULT_FILES[file_option]["ts"], lat, lon, color)
    return None

def setup_regionalization(df):
    """Sets up regionalization settings."""
    regionalize = st.sidebar.checkbox("Enable Regionalization")
    if not regionalize or df is None:
        return None, None, None
    
    with st.sidebar.expander("Regionalization Settings"):
        shp_source = st.selectbox("Shapefile Source", ["Use Default Shapefile", "Upload My Own Shapefile"])
        gdf = load_shapefile(shp_source, DEFAULT_SHAPEFILE)
        if gdf is None or gdf.empty:
            st.error("Failed to load shapefile or shapefile is empty.")
            return None, None, None
        region_col = st.selectbox("Region Column", gdf.columns, index=gdf.columns.get_loc("name") if "name" in gdf.columns else 0)
        region = st.selectbox("Region", ["Select a region"] + sorted(gdf[region_col].unique()))
        if region == "Select a region":
            return gdf, None, region_col
        bounds = get_region_bounds(gdf, region, region_col)
        if bounds:
            st.write(f"Region Bounds: Lon [{bounds['min_lon']:.5f}, {bounds['max_lon']:.5f}], Lat [{bounds['min_lat']:.5f}, {bounds['max_lat']:.5f}]")
        return gdf, region, region_col

# ------------------------------------
# Section E: Main Logic
# ------------------------------------
def main():
    """Runs the Streamlit app."""
    file_option, data_config, vis_config = setup_ui()
    
    # Load data and setup data settings
    df, file_type, file_path, file_name, columns = load_data(DEFAULT_FILES.get(file_option, {}), file_option, None)
    if df is None:
        st.info("Please select or upload a file.")
        return
    
    value_col, no_data_value, data_config = setup_data_settings(columns, file_option)
    df, file_type, file_path, file_name, columns = load_data(DEFAULT_FILES.get(file_option, {}), file_option, no_data_value)
    
    if df is None:
        st.info("Please select or upload a file.")
        return
    
    # Store the original column name for NetCDF access
    original_value_col = value_col
    
    # Rename selected column to 'value' for consistency in DataFrame
    if value_col and value_col != "value":
        df = df.rename(columns={value_col: "value"})
        value_col = "value"
    
    # Setup visualization and time series
    vis_config = setup_visualization_settings()
    ts_fig = setup_time_series(file_option)
    
    # Setup regionalization
    gdf, region, region_col = setup_regionalization(df)
    if region:
        df = filter_by_region(df, gdf, region, region_col)
    
    if data_config["show_raw"]:
        st.write("Raw Data:")
        st.dataframe(df)
    
    # Render visualizations
    tabs = st.tabs(["Heatmap", "3D Plot"] + (["Scatter Plot"] if file_type == "csv" else []))
    with tabs[0]:
        st.plotly_chart(plot_heatmap(df, file_name, vis_config, value_col), use_container_width=True)
    with tabs[1]:
        # Pass original_value_col for NetCDF access
        st.plotly_chart(plot_3d(df, file_type, file_path, file_name, vis_config, original_value_col, {"gdf": gdf, "region": region, "region_col": region_col}), use_container_width=True)
    if file_type == "csv" and len(tabs) > 2:
        with tabs[2]:
            st.plotly_chart(plot_scatter(df, file_name, vis_config, value_col), use_container_width=True)
    
    if ts_fig:
        st.plotly_chart(ts_fig, use_container_width=True)

if __name__ == "__main__":
    main()