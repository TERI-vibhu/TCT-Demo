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
    #df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    # Check if 'latitude' and 'longitude' exist, if not try 'lat' and 'lon'
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    elif 'lat' not in df.columns and 'lon' not in df.columns:
        st.error("Latitude/Longitude columns not found. Please ensure your NetCDF contains 'latitude' and 'longitude' or 'lat' and 'lon'.")
        return None, None  # Return None to prevent further processing
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
        height=1100,
        width=1000
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_netcdf_3d(data, lons_axis, lats_axis, lon_coord, lat_coord, selected_var, color_scale):
    """Plots a 3D surface of NetCDF data using Plotly."""

    fig = go.Figure(data=go.Surface(
        z=data,
        x=lons_axis,
        y=lats_axis,
        colorscale=color_scale,
        connectgaps=False,
        colorbar=dict(title=selected_var),
        hovertemplate=f"{lon_coord}: %{{x:.5f}}<br>{lat_coord}: %{{y:.5f}}<br>Value: %{{z}}<extra></extra>"
    ))
    fig.update_layout(
        title=f"{selected_var} - 3D Surface",
        scene=dict(xaxis_title=lon_coord, yaxis_title=lat_coord, zaxis_title=selected_var),
        height=1100, margin={"r":0,"t":40,"l":0,"b":0}
    )

    return fig


def plot_csv_3d(x, y, z, is_regular_grid, color_scale):
    """Plots a 3D surface or scatter plot of CSV data using Plotly."""
    fig = None
    if is_regular_grid and x is not None and y is not None and z is not None:
        # Use go.Surface for regular grid data.
        fig = go.Figure(data=go.Surface(
            z=z,
            x=x,
            y=y,
            colorscale=color_scale,
            connectgaps=False,  # Don't draw surfaces over missing data points
            hovertemplate="Longitude: %{x:.5f}<br>Latitude: %{y:.5f}<br>Value: %{z}<extra></extra>",
            colorbar=dict(title="Value")
        ))
        fig.update_layout(
            title="CSV Visualization - 3D Surface (Regular Grid)",
            scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Value"),
            height=1100, margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
    elif x is not None and y is not None and z is not None:  # Handle the irregular case
        # Use go.Scatter3d for irregular data (points in 3D space).
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=3, color=z, colorscale=color_scale, opacity=0.8, colorbar=dict(title="Value")),
            hovertemplate="Longitude: %{x:.5f}<br>Latitude: %{y:.5f}<br>Value: %{z}<extra></extra>"
        ))
        fig.update_layout(
            title="CSV Visualization - 3D Scatter (Irregular Data)",
            scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Value"),
            height=1100, margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
    else:
        st.error("Could not prepare data for 3D plot.")
    return fig

# Function to plot TIFF data in 3D
def plot_tiff_3d(data, lons_axis, lats_axis, color_scale):
    """Plots a 3D surface of TIFF data using Plotly."""
    fig = go.Figure(data=go.Surface(
        z=data,
        x=lons_axis,
        y=lats_axis,
        colorscale=color_scale,
        connectgaps=False,
        colorbar=dict(title="Value"),
        hovertemplate="Longitude: %{x:.5f}<br>Latitude: %{y:.5f}<br>Value: %{z}<extra></extra>"
    ))
    fig.update_layout(
        title="TIFF Visualization - 3D Surface",
        scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Value"),
        height=1100, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    return fig


# ------------------------------------
# Streamlit App
# ------------------------------------
st.title("TCT Demo")
st.sidebar.header("Controls")

# Example default file options
default_files = {
    "Rainfall 1994-2023 Netcdf": "IMD_RF_AVG_after1994.nc",
    #"Minimum Average Temperature": "gridpoint_temperature_stats_min.csv",
    #"Maximum Average Temperature": "gridpoint_temperature_stats_max.csv",
    "Rainfall 1994-2023 TIFF ": "IMD_RF_avg_1994-2023.tif",
    "Rainfall 1994-2023 CSV": "IMD_RF_AVG_after1994.csv"
}

data_source = st.sidebar.radio("Choose Data Source:", ["Use Default File", "Upload My Own File"])

df, file_type = None, None
tmp_file_path = None  # Initialize tmp_file_path

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
            # We'll delete the file later, after plotting
            pass

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

    # Visualization type selection
    if file_type in ["nc", "netcdf", "tif", "tiff"]:
        viz_type = st.sidebar.radio("Visualization Type", ["Grid Heatmap", "3D Surface"])
    elif file_type == "csv":
        viz_type = st.sidebar.radio("Visualization Type", ["Grid Heatmap", "3D Surface", "Scatter Plot"])

    if file_type == "csv":
        # CSV Specific Visualizations
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
                height=1100,
                width=1000
            )

            st.plotly_chart(fig, use_container_width=True)
        elif viz_type == "3D Surface":
            # Assuming CSV data might not always be on a regular grid.  Add a check.
            # You'll need to determine if your CSV data is a regular grid or not.
            # For this example, let's add a checkbox to the sidebar.
            is_regular_grid = st.sidebar.checkbox("Is CSV data on a regular grid?", value=False)
            fig = plot_csv_3d(df['lon'], df['lat'], df['value'], is_regular_grid, color_scale)
            if fig:
                st.plotly_chart(fig, use_container_width=True)  # Display the 3D plot if successfully created
        else:
            plot_heatmap(df, "value", map_style, color_scale)
    elif file_type in ["nc", "netcdf"]:
         # NetCDF Specific Visualizations
        if viz_type == "3D Surface":
            # Assuming your NetCDF data has 'lon' and 'lat' coordinates
            # You might need to adjust these based on the actual names in your file
            try:
                current_file_path = tmp_file_path if data_source == "Upload My Own File" else file_path
                ds = xr.open_dataset(current_file_path) # Re-open the dataset to get coordinates
                lons = ds['lon'].values
                lats = ds['lat'].values
                variable = list(ds.data_vars)[0]  # pick the first variable
                data = ds[variable].squeeze().values
                fig = plot_netcdf_3d(data,lats,lons,  "Latitude","Longitude", variable, color_scale)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating 3D plot for NetCDF: {e}")
        else:
            plot_heatmap(df, "value", map_style, color_scale)
    elif file_type in ["tif", "tiff"]:
        # TIFF Specific Visualizations
        if viz_type == "3D Surface":
            try:
                current_file_path = tmp_file_path if data_source == "Upload My Own File" else file_path
                with rasterio.open(current_file_path) as src:
                    # Read the original array - use this directly for visualization
                    array = src.read(1)
                    array = np.where(array == src.nodata, np.nan, array)
                    
                    # Get coordinates
                    bounds = src.bounds
                    res = src.res
                    lons = np.arange(bounds.left + res[0]/2, bounds.right, res[0])
                    lats = np.arange(bounds.top - res[1]/2, bounds.bottom, -res[1])
                
                # Use the original array for the 3D plot, not the reshaped df['value']
                fig = plot_tiff_3d(array, lats,lons, color_scale)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating 3D plot for TIFF: {e}")
        else:
            plot_heatmap(df, "value", map_style, color_scale)

    # Clean up temporary file if needed
    if tmp_file_path and os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
        
else:
    st.info("Please select or upload a file to begin.")