import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xarray as xr
import rasterio
from rasterio.plot import show
import tempfile
import os

# ------------------------------------
# Utility Functions
# ------------------------------------
def get_file_extension(file_path):
    return file_path.split('.')[-1].lower()

def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df, "csv"

def load_netcdf_data(file_path):
    ds = xr.open_dataset(file_path)
    variable = list(ds.data_vars)[0]  # pick the first variable
    df = ds[variable].squeeze().to_dataframe().reset_index()
    df = df.dropna()
    df = df.rename(columns={df.columns[-1]: "value"})
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    return df, "netcdf"

def load_tiff_data(file_path):
    with rasterio.open(file_path) as src:
        array = src.read(1)
        array = np.where(array == src.nodata, np.nan, array)
        bounds = src.bounds
        res = src.res

        lons = np.arange(bounds.left + res[0]/2, bounds.right, res[0])
        lats = np.arange(bounds.top - res[1]/2, bounds.bottom, -res[1])

        lon_grid, lat_grid = np.meshgrid(lons, lats)

        df = pd.DataFrame({
            "lon": lon_grid.flatten(),
            "lat": lat_grid.flatten(),
            "value": array.flatten()
        })
        df = df.dropna()
    return df, "tiff"

def calculate_zoom(lat_range, lon_range):
    max_range = max(lat_range, lon_range)
    if max_range > 10:
        return 3.5
    elif max_range > 5:
        return 4
    elif max_range > 1:
        return 6
    elif max_range > 0.1:
        return 8
    else:
        return 11

def plot_heatmap(df, values, map_style, color_scale):
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    zoom = int(calculate_zoom(df['lat'].max() - df['lat'].min(), df['lon'].max() - df['lon'].min()))

    # Determine default grid size
    try:
        x_diff = np.min(np.diff(np.sort(df['lon'].unique())))
        y_diff = np.min(np.diff(np.sort(df['lat'].unique())))
    except:
        x_diff, y_diff = 0.1, 0.1

    # Sidebar advanced options
    with st.sidebar.expander("Advanced Settings", expanded=False):
        cell_width = st.number_input("Cell Width (longitude)", value=float(x_diff), step=0.001)
        cell_height = st.number_input("Cell Height (latitude)", value=float(y_diff), step=0.001)
        grid_opacity = st.number_input("Grid Opacity", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

    features = []
    for i, row in df.iterrows():
        lon = row['lon']
        lat = row['lat']
        val = row['value']

        feature = {
            "type": "Feature",
            "id": i,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon - cell_width/2, lat - cell_height/2],
                    [lon + cell_width/2, lat - cell_height/2],
                    [lon + cell_width/2, lat + cell_height/2],
                    [lon - cell_width/2, lat + cell_height/2],
                    [lon - cell_width/2, lat - cell_height/2]
                ]]
            },
            "properties": {
                "value": val,
                "lat": lat,
                "lon": lon
            }
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}
    locations = [f["id"] for f in features]
    color_values = [f["properties"]["value"] for f in features]
    hover_lats = [f["properties"]["lat"] for f in features]
    hover_lons = [f["properties"]["lon"] for f in features]

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=locations,
        z=color_values,
        colorscale=color_scale,
        marker_opacity=grid_opacity,
        marker_line_width=0,
        colorbar=dict(title="Value"),
        customdata=np.stack((hover_lats, hover_lons), axis=-1),
        hovertemplate="<b>Lat: %{customdata[0]:.5f}</b><br>Lon: %{customdata[1]:.5f}<br>Value: %{z}<extra></extra>"
    ))

    fig.update_layout(
        mapbox=dict(
            style=map_style,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        height=700,
        width=1000
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------
# Streamlit App
# ------------------------------------
st.title("TCT Demo")
st.sidebar.header("Controls")

# Example default file options
default_files = {
    "Rainfall 1994-2023 Netcdf": "IMD_RF_AVG_after1994.nc",
    "Minimum Average Temperature": "gridpoint_temperature_stats_min.csv",
    "Maximum Average Temperature": "gridpoint_temperature_stats_max.csv",
    #"Rainfall 1994-2023 Netcdf": "IMD_RF_AVG_after1994.nc",
    #"Rainfall 1994-2023 TIFF ": "IMD_RF_avg_1994-2023.tif"
}

'''
default_files = {
    "Rainfall 1994-2023 Netcdf": "/home/vibhu/Downloads/TCT/IMD_RF_AVG_after1994.nc",
    "Minimum Average Temperature": "/home/vibhu/Downloads/TCT/gridpoint_temperature_stats_min.csv",
    "Maximum Average Temperature": "/home/vibhu/Downloads/TCT/gridpoint_temperature_stats_max.csv"
 
    #"Rainfall 1994-2023 TIFF ": "/home/vibhu/Downloads/TCT/IMD_RF_avg_1994-2023.tif"
    #"Example TIFF": "path/to/example3.tif"
}
'''
data_source = st.sidebar.radio("Choose Data Source:", ["Use Default File", "Upload My Own File"])

df, file_type = None, None

if data_source == "Use Default File":
    selected_file = st.sidebar.selectbox("Select a file:", list(default_files.keys()))
    file_path = default_files[selected_file]
    ext = get_file_extension(file_path)

    if ext == "csv":
        df, file_type = load_csv_data(file_path)
    elif ext in ["nc", "netcdf"]:
        df, file_type = load_netcdf_data(file_path)
    elif ext in ["tif", "tiff"]:
        df, file_type = load_tiff_data(file_path)
    else:
        st.sidebar.error("Unsupported file type.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "nc", "netcdf", "tif", "tiff"])
    if uploaded_file is not None:
        ext = get_file_extension(uploaded_file.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        try:
            if ext == "csv":
                df, file_type = load_csv_data(tmp_file_path)
            elif ext in ["nc", "netcdf"]:
                df, file_type = load_netcdf_data(tmp_file_path)
            elif ext in ["tif", "tiff"]:
                df, file_type = load_tiff_data(tmp_file_path)
            else:
                st.sidebar.error("Unsupported file type.")
        finally:
            os.remove(tmp_file_path)

# ------------------------------------
# Visualization
# ------------------------------------
if df is not None:
    st.sidebar.subheader("Visualization Settings")
    show_raw = st.sidebar.checkbox("Show Raw Data", value=False)

    if show_raw:
        st.write(df)

    map_style = st.sidebar.selectbox("Map Style", ["carto-positron","open-street-map" , "stamen-terrain"])
    color_scale = st.sidebar.selectbox("Color Scale", ["Blues","Viridis", "Plasma", "Inferno", "Cividis", "Reds"])

    if file_type == "csv":
        viz_type = st.sidebar.radio("Visualization Type", ["Grid Heatmap", "Scatter Plot"])
        value_col = st.sidebar.selectbox("Value Column", [col for col in df.columns if col not in ["lon", "lat"]])
        df = df.rename(columns={value_col: "value"})

        if viz_type == "Scatter Plot":
            fig = go.Figure(go.Scattermapbox(
                lat=df['lat'],
                lon=df['lon'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['value'],
                    colorscale=color_scale,
                    opacity=0.7,
                    colorbar=dict(title="Value"),
                ),
                hovertemplate="<b>Lat: %{lat:.5f}</b><br>Lon: %{lon:.5f}<br>Value: %{marker.color}<extra></extra>"
            ))

            fig.update_layout(
                mapbox=dict(
                    style=map_style,
                    center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()),
                    zoom=int(calculate_zoom(df['lat'].max() - df['lat'].min(), df['lon'].max() - df['lon'].min()))
                ),
                height=700,
                width=1000
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            plot_heatmap(df, "value", map_style, color_scale)
    else:
        plot_heatmap(df, "value", map_style, color_scale)
else:
    st.info("Please select or upload a file to begin.")
