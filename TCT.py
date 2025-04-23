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

# Set Streamlit page configuration
st.set_page_config(
    page_title='TERI Climate Tool Demo',
    page_icon='🌏',
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

st.logo("data/TERI Logo Seal.png",size='large')  # Add your logo path here
# ------------------------------------
# Section A: Utility Functions
# ------------------------------------
def get_file_extension(file_path):
    """Extracts the file extension from a file path."""
    return file_path.split('.')[-1].lower()

def load_csv_data(file_path, no_data_value):
    """Loads CSV data into a DataFrame and filters no-data values."""
    df = pd.read_csv(file_path)
    if no_data_value is not None:
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].where(df[col] != no_data_value, np.nan)
    df = df.dropna()
    return df, "csv"

def load_netcdf_data(file_path, no_data_value, selected_variable=None):
    """Loads NetCDF data, filters no-data values, and converts it to a DataFrame."""
    ds = xr.open_dataset(file_path)
    available_vars = list(ds.data_vars)
    if not available_vars:
        st.error("No variables found in NetCDF file.")
        return None, None, []
    # If no variable is selected, use the first one
    variable = selected_variable if selected_variable in available_vars else available_vars[0]
    if no_data_value is not None:
        ds[variable] = ds[variable].where(ds[variable] != no_data_value, np.nan)
    df = ds[variable].squeeze().to_dataframe().reset_index()
    df = df.dropna()
    df = df.rename(columns={df.columns[-1]: "value"})
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    elif 'lat' not in df.columns and 'lon' not in df.columns:
        st.error("Latitude/Longitude columns not found.")
        return None, None, []
    return df, "netcdf", available_vars

def load_tiff_data(file_path, no_data_value):
    """Loads TIFF data, filters no-data values, and converts it to a DataFrame."""
    with rasterio.open(file_path) as src:
        array = src.read(1)
        # Use user-specified no-data value if provided, else use file's nodata
        nodata = no_data_value if no_data_value is not None else src.nodata
        array = np.where(array == nodata, np.nan, array)
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
    return df, "tiff", ["value"]

def load_shapefile(shp_path):
    """Loads a shapefile using geopandas."""
    try:
        gdf = gpd.read_file(shp_path)
        return gdf
    except Exception as e:
        st.error(f"Error loading shapefile: {e}")
        return None

def filter_data_by_region(df, gdf, selected_region, region_column):
    """Filters DataFrame to include only points within the selected region's geometry."""
    if gdf is None or selected_region is None:
        return df
    try:
        region_geometry = gdf[gdf[region_column] == selected_region].geometry.iloc[0]
        geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
        gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs=gdf.crs)
        filtered_gdf = gdf_points[gdf_points.geometry.within(region_geometry)]
        filtered_df = filtered_gdf.drop(columns='geometry').reset_index(drop=True)
        return filtered_df
    except Exception as e:
        st.error(f"Error filtering data by region: {e}")
        return df

def get_region_bounds(gdf, selected_region, region_column):
    """Returns the latitude and longitude bounds of the selected region."""
    if gdf is None or selected_region is None:
        return None
    try:
        region_geometry = gdf[gdf[region_column] == selected_region].geometry.iloc[0]
        minx, miny, maxx, maxy = region_geometry.bounds
        return {"min_lon": minx, "max_lon": maxx, "min_lat": miny, "max_lat": maxy}
    except Exception as e:
        st.error(f"Error calculating region bounds: {e}")
        return None

def calculate_zoom(lat_range, lon_range):
    """Calculates the map zoom level based on latitude and longitude ranges."""
    max_range = max(lat_range, lon_range)
    if max_range > 10:
        return 4
    elif max_range > 5:
        return 5
    elif max_range > 1:
        return 6
    elif max_range > 0.1:
        return 8
    else:
        return 11

def read_netcdf_ts(file_path, lat, lon, selected_variable=None, line_color='blue'):
    """Reads NetCDF time series data for a specific lat/lon and creates a time series plot."""
    try:
        ds = xr.open_dataset(file_path)
        available_vars = list(ds.data_vars)
        if not available_vars:
            st.error("No variables found in time series NetCDF file.")
            return None
        
        # Use selected variable or first available
        variable = selected_variable if selected_variable in available_vars else available_vars[0]
        
        # Find nearest lat/lon point
        lat_values = ds['lat'].values if 'lat' in ds.coords else ds['latitude'].values
        lon_values = ds['lon'].values if 'lon' in ds.coords else ds['longitude'].values
        lat_idx = np.abs(lat_values - lat).argmin()
        lon_idx = np.abs(lon_values - lon).argmin()
        
        # Extract time series
        ts_data = ds[variable].isel(
            lat=lat_idx,
            lon=lon_idx
        ).to_dataframe().reset_index()
        
        # Create time series plot with user-selected color
        fig = px.line(
            ts_data,
            x='year',
            y=variable,
            title=f'Time Series at Lat: {lat:.2f}, Lon: {lon:.2f}',
            labels={'time': 'Time', variable: 'Value'},
            color_discrete_sequence=[line_color]
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=variable,
            height=600,
            width=1000
        )
        return fig
    except Exception as e:
        st.error(f"Error processing time series data: {e}")
        return None

# ------------------------------------
# Section B: Visualization Functions
# ------------------------------------
def plot_heatmap(df, file_name, cell_width, cell_height, grid_opacity, map_style, color_scale):
    """Creates a heatmap visualization using Plotly Choroplethmapbox."""
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    zoom = int(calculate_zoom(df['lat'].max() - df['lat'].min(), df['lon'].max() - df['lon'].min()))

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
        height=900,
        width=1000,
        title=f"Grid Heatmap - {file_name}"
    )
    return fig

def plot_netcdf_3d(data, lons_axis, lats_axis, lon_coord, lat_coord, selected_var, color_scale, file_name):
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
        title=f"3D Surface Plot - {file_name}",
        scene=dict(xaxis_title=lon_coord, yaxis_title=lat_coord, zaxis_title=selected_var),
        height=900, margin={"r":0,"t":40,"l":0,"b":0}
    )
    return fig

def plot_csv_3d(x, y, z, is_regular_grid, color_scale, file_name):
    """Plots a 3D surface or scatter plot of CSV data using Plotly."""
    fig = None
    if is_regular_grid and x is not None and y is not None and z is not None:
        fig = go.Figure(data=go.Surface(
            z=z,
            x=x,
            y=y,
            colorscale=color_scale,
            connectgaps=False,
            hovertemplate="Longitude: %{x:.5f}<br>Latitude: %{y:.5f}<br>Value: %{z}<extra></extra>",
            colorbar=dict(title="Value")
        ))
        fig.update_layout(
            title=f"3D Surface - {file_name}",
            scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Value"),
            height=900, margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
    elif x is not None and y is not None and z is not None:
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=3, color=z, colorscale=color_scale, opacity=0.8, colorbar=dict(title="Value")),
            hovertemplate="Longitude: %{x:.5f}<br>Latitude: %{y:.5f}<br>Value: %{z}<extra></extra>"
        ))
        fig.update_layout(
            title=f"3D Scatter - {file_name}",
            scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Value"),
            height=900, margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
    return fig

def plot_tiff_3d(data, lons_axis, lats_axis, color_scale, file_name):
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
        title=f"3D Surface - {file_name}",
        scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Value"),
        height=900, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    return fig

def plot_csv_scatter(df, map_style, color_scale, file_name):
    """Plots a scatter plot for CSV data using Plotly Scattermapbox."""
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
        height=900,
        width=1000,
        title=f"Scatter Plot - {file_name}"
    )
    return fig

# ------------------------------------
# Section C: Data Loading Functions
# ------------------------------------
def load_data_file(default_files, selected_option):
    """Loads data file based on user selection and allows value column selection."""
    df, file_type, tmp_file_path, selected_file, uploaded_file_name, available_cols, value_col, no_data_value = None, None, None, None, None, [], None, None
    
    # Initialize no-data value based on selection
    if selected_option != "Upload Your Own Data" and selected_option in default_files:
        default_no_data = default_files[selected_option].get('no_data', -9999)
    else:
        default_no_data = -9999  # Default for uploaded files
    
    if selected_option == "Upload Your Own Data":
        uploaded_file = st.sidebar.file_uploader("Upload Your Data File", type=["csv", "nc", "netcdf", "tif", "tiff"], key="data_file_uploader")
        if uploaded_file is not None:
            uploaded_file_name = uploaded_file.name
            ext = get_file_extension(uploaded_file.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            if ext == "csv":
                df, file_type = load_csv_data(tmp_file_path, no_data_value)
                available_cols = [col for col in df.columns if col not in ["lon", "lat"]]
            elif ext in ["nc", "netcdf"]:
                df, file_type, available_cols = load_netcdf_data(tmp_file_path, no_data_value)
            elif ext in ["tif", "tiff"]:
                df, file_type, available_cols = load_tiff_data(tmp_file_path, no_data_value)
            else:
                st.sidebar.error("Unsupported file type.")
    else:
        selected_file = selected_option
        file_path = default_files[selected_file]['avg']
        ext = get_file_extension(file_path)
        if ext == "csv":
            df, file_type = load_csv_data(file_path, no_data_value)
            available_cols = [col for col in df.columns if col not in ["lon", "lat"]]
        elif ext in ["nc", "netcdf"]:
            df, file_type, available_cols = load_netcdf_data(file_path, no_data_value)
        elif ext in ["tif", "tiff"]:
            df, file_type, available_cols = load_tiff_data(file_path, no_data_value)
        else:
            st.sidebar.error("Unsupported file type.")
    
    # Advanced Data Settings
    with st.sidebar.expander("Advanced Data Settings", expanded=True):
        # Variable Selection
        if available_cols:
            value_col = st.selectbox("Select Variable", available_cols, key="value_column")
        else:
            st.write("No variables available to select.")
        
        no_data_value = st.number_input(
            "No-Data Value",
            value=default_no_data,
            step=1,
            format="%d",
            key="no_data_value_input"
        )
        show_raw = st.checkbox("Show Raw Data", value=False, key="show_raw_checkbox")
        enable_regionalization = st.checkbox("Enable Data Regionalization", value=False, key="enable_regionalization_checkbox")
    
    return df, file_type, tmp_file_path, selected_file, uploaded_file_name, available_cols, value_col, no_data_value, show_raw, enable_regionalization

def load_shapefile_data(shp_source, default_shapefile):
    """Loads shapefile based on user selection."""
    shp_gdf, shp_tmp_file_path = None, None
    if shp_source == "Use Default Shapefile":
        if os.path.exists(default_shapefile):
            shp_gdf = load_shapefile(default_shapefile)
        else:
            st.error("Default shapefile not found. Please upload a shapefile.")
    else:
        uploaded_shp = st.file_uploader("Upload Your Shapefile", type=["shp"])
        if uploaded_shp is not None:
            shp_tmp_dir = tempfile.mkdtemp()
            shp_base_name = os.path.splitext(uploaded_shp.name)[0]
            shp_tmp_file_path = os.path.join(shp_tmp_dir, uploaded_shp.name)
            with open(shp_tmp_file_path, "wb") as f:
                f.write(uploaded_shp.read())
            associated_extensions = ['.shx', '.dbf', '.prj', '.cpg']
            for ext in associated_extensions:
                st.file_uploader(f"Upload {ext} file for shapefile (if required)", type=[ext], key=f"shp_{ext}")
                uploaded_assoc = st.session_state.get(f"shp_{ext}")
                if uploaded_assoc:
                    assoc_path = os.path.join(shp_tmp_dir, f"{shp_base_name}{ext}")
                    with open(assoc_path, "wb") as f:
                        f.write(uploaded_assoc.read())
            shp_gdf = load_shapefile(shp_tmp_file_path)
    return shp_gdf, shp_tmp_file_path

# ------------------------------------
# Section D: Streamlit App Setup
# ------------------------------------
def setup_ui():
    """Sets up the Streamlit app UI."""
    st.markdown("<h1 style='color: #dc6142';'>TERI Climate Tool : Demo </h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='color: #dc6142';'>Data and Visualization Controls</h2>", unsafe_allow_html=True)

    # Default file options with no-data values
    default_files = {
        "Rainfall 1994-2023": {
            "avg": "data/IMD_RF_AVG_after1994.nc",
            "ts": "data/RF_yearly.nc",
            "no_data": -999 # Default no-data value for NetCDF
        },
        "Rainfall 1994-2023 TIFF": {
            "avg": "data/IMD_RF_avg_1994-2023.tif",
            "ts": None,
            "no_data": -9999  # Default no-data value for TIFF
        },
        "Rainfall 1994-2023 CSV": {
            "avg": "data/IMD_RF_AVG_after1994.csv",
            "ts": None,
            "no_data": -9999  # Default no-data value for CSV
        }
    }
    default_shapefile = "data/India_State_Boundary.shp"  # Replace with actual path

    return default_files, default_shapefile

def setup_time_series_settings(default_files, selected_file):
    """Sets up time series settings in the sidebar."""
    ts_fig = None
    # Add latitude/longitude inputs and color selection if time series file exists
    if selected_file and default_files[selected_file].get('ts'):
        with st.sidebar.expander("Advanced Time Series Settings", expanded=False):
            lat_input = st.number_input("Enter Latitude", min_value=-90.0, max_value=90.0, value=28.5, step=0.5, key="lat_input")
            lon_input = st.number_input("Enter Longitude", min_value=-180.0, max_value=180.0, value=77.5, step=0.5, key="lon_input")
            # Color options supported by Plotly
            line_color = st.selectbox(
                "Line Color",
                ["blue", "red", "green", "purple", "orange", "black", "cyan", "magenta"],
                index=0,
                key="ts_line_color"
            )
            if st.button("Generate Time Series"):
                ts_fig = read_netcdf_ts(default_files[selected_file]['ts'], lat_input, lon_input, line_color=line_color)
    return ts_fig

def setup_regionalization(df, default_shapefile, enable_regionalization):
    """Sets up regionalization feature."""
    shp_gdf, shp_tmp_file_path, selected_region, region_column = None, None, None, "name"
    
    if enable_regionalization and df is not None:
        with st.sidebar.expander("Regionalization Settings", expanded=True):
            shp_source = st.selectbox("Shapefile Source:", ["Use Default Shapefile", "Upload My Own Shapefile"])
            shp_gdf, shp_tmp_file_path = load_shapefile_data(shp_source, default_shapefile)
            
            if shp_gdf is not None:
                region_column = st.selectbox("Select Column for Region Names:", shp_gdf.columns, 
                                            index=shp_gdf.columns.get_loc("name") if "name" in shp_gdf.columns else 0)
                region_names = sorted(shp_gdf[region_column].unique())
                selected_region = st.selectbox("Select Region:", ["Select a region"] + region_names)
                if selected_region == "Select a region":
                    selected_region = None
                else:
                    bounds = get_region_bounds(shp_gdf, selected_region, region_column)
                    if bounds:
                        st.write("Selected Region Bounds:")
                        st.write(f"Longitude: [{bounds['min_lon']:.5f}, {bounds['max_lon']:.5f}]")
                        st.write(f"Latitude: [{bounds['min_lat']:.5f}, {bounds['max_lat']:.5f}]")
    
    return shp_gdf, shp_tmp_file_path, selected_region, region_column

def setup_visualization_settings(df):
    """Sets up visualization settings in the sidebar."""
    try:
        x_diff = np.min(np.diff(np.sort(df['lon'].unique())))
        y_diff = np.min(np.diff(np.sort(df['lat'].unique())))
    except:
        x_diff, y_diff = 0.1, 0.1

    with st.sidebar.expander("Advanced Visualization Settings", expanded=False):
        cell_width = st.number_input("Cell Width (longitude)", value=float(x_diff), step=0.001, key="heatmap_width")
        cell_height = st.number_input("Cell Height (latitude)", value=float(y_diff), step=0.001, key="heatmap_height")
        grid_opacity = st.number_input("Grid Opacity", min_value=0.0, max_value=1.0, value=0.8, step=0.05, key="heatmap_opacity")
        map_style = st.selectbox("Map Style", ["carto-positron", "open-street-map", "stamen-terrain"], key="heatmap_map_style")
        color_scale = st.selectbox("Color Scale", ["Blues", "Viridis", "Plasma", "Inferno", "Cividis", "Reds"], key="heatmap_color_scale")
    
    return cell_width, cell_height, grid_opacity, map_style, color_scale

# ------------------------------------
# Section E: Main Visualization Logic
# ------------------------------------
def render_visualizations(df, file_type, value_col, tmp_file_path, 
                         enable_regionalization, shp_gdf, selected_region, region_column, show_raw,
                         default_files, selected_file, ts_fig, cell_width, cell_height, grid_opacity, 
                         map_style, color_scale, uploaded_file_name=None):
    """Renders visualizations using tabs."""
    # Determine file name to display
    file_name = selected_file if selected_file else uploaded_file_name if uploaded_file_name else "Unknown File"

    if show_raw:
        st.write("Raw Data Preview:")
        st.dataframe(df)

    # Filter data by region if enabled
    if enable_regionalization and selected_region is not None:
        df = filter_data_by_region(df, shp_gdf, selected_region, region_column)

    # Ensure the selected value column is renamed to 'value'
    if value_col and value_col != "value":
        df = df.rename(columns={value_col: "value"})

    # Setup tabs
    tabs = ["Heatmap", "3D Plot"] if file_type in ["netcdf", "tiff"] else ["Heatmap", "3D Plot", "Scatter Plot"]
    heatmap_tab, threed_tab, *scatter_tab = st.tabs(tabs)

    # Heatmap Tab
    with heatmap_tab:
        fig = plot_heatmap(df, file_name, cell_width, cell_height, grid_opacity, map_style, color_scale)
        st.plotly_chart(fig, use_container_width=True)

    # 3D Plot Tab
    with threed_tab:
        if file_type == "csv":
            is_regular_grid = st.checkbox("Is Csv data on a regular grid?", value=False, key="csv_grid")
            fig = plot_csv_3d(df['lon'], df['lat'], df['value'], is_regular_grid, color_scale, file_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not prepare data for 3D plot.")
        elif file_type == "netcdf":
            try:
                current_file_path = tmp_file_path if tmp_file_path else default_files[selected_file]['avg'] if selected_file else None
                if not current_file_path:
                    st.error("No valid file path provided for NetCDF 3D plot.")
                    return
                ds = xr.open_dataset(current_file_path)
                lons = ds['lon'].values
                lats = ds['lat'].values
                data = ds[value_col].squeeze().values  # Use selected value_col
                if enable_regionalization and selected_region is not None:
                    bounds = get_region_bounds(shp_gdf, selected_region, region_column)
                    if bounds:
                        lon_mask = (lons >= bounds['min_lon']) & (lons <= bounds['max_lon'])
                        lat_mask = (lats >= bounds['min_lat']) & (lats <= bounds['max_lat'])
                        lons = lons[lon_mask]
                        lats = lats[lat_mask]
                        data = data[np.ix_(lat_mask, lon_mask)]
                fig = plot_netcdf_3d(data, lons, lats, "Longitude", "Latitude", value_col, color_scale, file_name)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating 3D plot for NetCDF: {e}")
        elif file_type == "tiff":
            try:
                current_file_path = tmp_file_path if tmp_file_path else default_files[selected_file]['avg'] if selected_file else None
                if not current_file_path:
                    st.error("No valid file path provided for TIFF 3D plot.")
                    return
                with rasterio.open(current_file_path) as src:
                    array = src.read(1)
                    array = np.where(array == src.nodata, np.nan, array)
                    bounds = src.bounds
                    res = src.res
                    lons = np.arange(bounds.left + res[0]/2, bounds.right, res[0])
                    lats = np.arange(bounds.top - res[1]/2, bounds.bottom, -res[1])
                    if enable_regionalization and selected_region is not None:
                        region_bounds = get_region_bounds(shp_gdf, selected_region, region_column)
                        if region_bounds:
                            lon_mask = (lons >= region_bounds['min_lon']) & (lons <= region_bounds['max_lon'])
                            lat_mask = (lats >= region_bounds['min_lat']) & (lats <= region_bounds['max_lat'])
                            lons = lons[lon_mask]
                            lats = lats[lat_mask]
                            array = array[np.ix_(lat_mask, lon_mask)]
                fig = plot_tiff_3d(array, lons, lats, color_scale, file_name)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating 3D plot for TIFF: {e}")

    # Scatter Plot Tab (CSV only)
    if file_type == "csv" and scatter_tab:
        with scatter_tab[0]:
            fig = plot_csv_scatter(df, map_style, color_scale, file_name)
            st.plotly_chart(fig, use_container_width=True)

    # Display Time Series Plot if Generated
    if ts_fig:
        st.plotly_chart(ts_fig, use_container_width=True)

# ------------------------------------
# Section F: Main App Logic
# ------------------------------------
def main():
    """Main function to run the Streamlit app."""
    default_files, default_shapefile = setup_ui()
    
    # 1. Default File Selection
    selected_option = st.sidebar.selectbox(
        "Select a Default File:",
        list(default_files.keys()) + ["Upload Your Own Data"],
        key="file_select"
    )
    
    # Load data file based on selection
    df, file_type, tmp_file_path, selected_file, uploaded_file_name, available_cols, value_col, no_data_value, show_raw, enable_regionalization = load_data_file(
        default_files=default_files,
        selected_option=selected_option
    )
    
    # 2. Advanced Data Settings (includes Variable Selection, handled in load_data_file)
    
    # 3. Advanced Visualization Settings
    cell_width, cell_height, grid_opacity, map_style, color_scale = setup_visualization_settings(df) if df is not None else (0.1, 0.1, 0.8, "carto-positron", "Blues")
    
    # 4. Advanced Time Series Settings
    ts_fig = setup_time_series_settings(default_files, selected_file)
    
    if df is not None:
        # Setup regionalization (only if enabled)
        shp_gdf, shp_tmp_file_path, selected_region, region_column = None, None, None, "name"
        if enable_regionalization:
            shp_gdf, shp_tmp_file_path, selected_region, region_column = setup_regionalization(
                df, default_shapefile, enable_regionalization
            )
        
        # Render visualizations
        render_visualizations(df, file_type, value_col, tmp_file_path, 
                             enable_regionalization, shp_gdf, selected_region, region_column, show_raw,
                             default_files, selected_file, ts_fig, cell_width, cell_height, 
                             grid_opacity, map_style, color_scale, uploaded_file_name)
        
        # Clean up temporary files
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        if shp_tmp_file_path and os.path.exists(shp_tmp_file_path):
            os.remove(shp_tmp_file_path)
    else:
        st.info("Please select or upload a file to begin.")

if __name__ == "__main__":
    main()