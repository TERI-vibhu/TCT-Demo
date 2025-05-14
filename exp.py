import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xarray as xr
import solara
from uuid import uuid4
from sklearn.neighbors import NearestNeighbors
import solara.lab  # Import for ThemeToggle
from scipy.stats import linregress
import geopandas as gpd  # Added for shapefile handling

# ------------------------------------
# Section A: Configuration
# ------------------------------------
DEFAULT_FILE_ID = "Rainfall Observed"  # Define default file ID explicitly
DEFAULT_SHAPEFILE_PATH = "/home/vibhu/venv/TCT/bin/data/India_State_Boundary.shp"  # Path to the shapefile

DEFAULT_FILE_ID = "Rainfall Observed"  # Define default file ID explicitly

FILE_CONFIG = [
    {
        "id": "Rainfall Observed",
        "name": "Rainfall Observed",
        "description": "Rainfall data from IMD 30 Year",
        "ts": "data/RF_yearly.nc",
        "avg": "data/IMD_RF_AVG_after1994.nc",
        "no_data": -999,
        "default_color_scale": "Blues"
    },
    {
        "id": "Minimum Temperature Projected Delta",
        "name": "Minimum Temperature Projected Delta",
        "description": "Minimum temperature projected delta from IMD",
        "ts": "data/Tmax-2081-2100_yearly.nc",
        "avg": "data/Tmin_1980-2100.nc",
        "no_data": 99.9,
        "default_color_scale": "Reds"
    },
    {
        "id": "Maximum Temperature Projected Delta",
        "name": "Maximum Temperature Projected Delta",
        "description": "Maximum temperature projected delta from IMD",
        "ts": "data/Tmax-2081-2100_yearly.nc",
        "avg": "data/Tmax_1980-2100.nc",
        "no_data": 99.9,
        "default_color_scale": "Reds"
    }
]

# Define available options for dropdowns
COLOR_SCALE_OPTIONS = ["Blues", "Viridis", "Reds", "Greens", "Inferno", "Magma"]
MAP_STYLE_OPTIONS = ["carto-positron", "carto-darkmatter", "open-street-map", "stamen-terrain"]

# Default values
DEFAULT_MAP_STYLE = "carto-positron"
DEFAULT_OPACITY = 0.8
DEFAULT_TS_COLOR = "blue"

# ------------------------------------
# Section B: Utility Functions
# ------------------------------------
def load_netcdf_data(file_config, no_data_value):
    """Loads NetCDF data, converts it to a DataFrame, and estimates grid spacing."""
    file_path = file_config["avg"]
    ds = xr.open_dataset(file_path)
    available_vars = list(ds.data_vars)
    float_vars = [var for var in available_vars if ds[var].dtype in ["float32", "float64"]]
    if not float_vars:
        raise ValueError("No float-type variables found in the NetCDF file.")
    
    df = ds.to_dataframe().reset_index().dropna()
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("NetCDF file must contain 'lat' and 'lon' coordinates.")
    
    # Estimate grid spacing using KNN
    if len(df) >= 4:  # Ensure enough points for sampling
        sample_indices = np.random.choice(len(df), size=4, replace=False)
        sample_coords = df[["lat", "lon"]].iloc[sample_indices].values
        coords = df[["lat", "lon"]].values
        
        knn = NearestNeighbors(n_neighbors=2)
        knn.fit(coords)
        
        distances, _ = knn.kneighbors(sample_coords, n_neighbors=2)
        avg_distance = distances[:, 1].mean()  # Distance to the second neighbor
    else:
        avg_distance = 0.25  # Fallback value
    
    return df, file_path, float_vars, avg_distance

def calculate_zoom(lat_range, lon_range):
    r = max(lat_range, lon_range)
    return 1 if r > 100 else 2 if r > 30 else 4 if r > 15 else 5 if r > 5 else 7 if r > 1 else 10 if r > 0.1 else 13

def plot_heatmap(df, file_name, value_col, cell_dimensions, color_scale, map_style, opacity, shapefile_path=DEFAULT_SHAPEFILE_PATH):
    """Creates a heatmap using Plotly Choroplethmap with a map and shapefile overlay."""
    CELL_HEIGHT = cell_dimensions
    CELL_WIDTH = cell_dimensions
    center = {"lat": df["lat"].mean(), "lon": df["lon"].mean()}
    zoom = calculate_zoom(df["lat"].max() - df["lat"].min(), df["lon"].max() - df["lon"].min())
    
    features = [
        {
            "type": "Feature", "id": str(uuid4()),
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [row["lon"] - CELL_WIDTH/2, row["lat"] - CELL_HEIGHT/2],
                    [row["lon"] + CELL_WIDTH/2, row["lat"] - CELL_HEIGHT/2],
                    [row["lon"] + CELL_WIDTH/2, row["lat"] + CELL_HEIGHT/2],
                    [row["lon"] - CELL_WIDTH/2, row["lat"] + CELL_HEIGHT/2],
                    [row["lon"] - CELL_WIDTH/2, row["lat"] - CELL_HEIGHT/2]
                ]]
            },
            "properties": {"value": row[value_col], "lat": row["lat"], "lon": row["lon"]}
        }
        for i, row in df.iterrows()
    ]
    
    geojson = {"type": "FeatureCollection", "features": features}
    fig = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            locations=[f["id"] for f in features],
            z=[f["properties"]["value"] for f in features],
            colorscale=color_scale,
            marker_opacity=opacity,
            marker_line_width=0,
            colorbar=dict(title=value_col),
            customdata=np.stack(([f["properties"]["lat"] for f in features], [f["properties"]["lon"] for f in features]), axis=-1),
            hovertemplate="<b>Lat: %{customdata[0]:.5f}</b><br>Lon: %{customdata[1]:.5f}<br>Value: %{z}<extra></extra>"
        )
    )

    # Overlay shapefile boundaries
    if shapefile_path:
        try:
            gdf = gpd.read_file(shapefile_path)
            for geom in gdf.geometry:
                if geom.geom_type == "Polygon":
                    coords = [geom.exterior.coords]
                elif geom.geom_type == "MultiPolygon":
                    coords = [poly.exterior.coords for poly in geom.geoms]
                else:
                    continue  # Skip non-polygon geometries
                
                for ring in coords:
                    lons, lats = zip(*ring)
                    fig.add_trace(
                        go.Scattergeo(
                            lon=lons,
                            lat=lats,
                            mode="lines",
                            line=dict(width=1, color="black"),
                            opacity=0.7,
                            showlegend=False,
                            hoverinfo="skip"
                        )
                    )
        except Exception as e:
            print(f"Error loading shapefile: {e}")

    fig.update_layout(
        map=dict(
            style=map_style,
            center=center,
            zoom=zoom
        ),
        height=900,
        width=1000,
        title=f"Heatmap - {file_name}"
    )
    return fig

def read_time_series(file_path, lat, lon, color):
    """Reads NetCDF time series data, creates a line plot with trend line, and annotates slope, p-value, and trend direction."""
    ds = xr.open_dataset(file_path)
    var = list(ds.data_vars)[0]
    
    # Find nearest lat/lon indices
    lat_idx = np.abs((ds.get("lat", ds.get("latitude")).values - lat)).argmin()
    lon_idx = np.abs((ds.get("lon", ds.get("longitude")).values - lon)).argmin()
    
    # Extract time series data
    ts_data = ds[var].isel(lat=lat_idx, lon=lon_idx).to_dataframe().reset_index()
    
    # Perform linear regression
    x = ts_data["year"].values
    y = ts_data[var].values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Determine trend direction
    trend_direction = "Increasing" if p_value < 0.05 and slope > 0 else "Decreasing" if p_value < 0.05 else "No Significant Trend"
    
    # Calculate trend line
    trend_line = intercept + slope * x
    ts_data["trend"] = trend_line
    
    # Create line plot with original data
    fig = px.line(
        ts_data,
        x="year",
        y=var,
        title=f"Time Series at Lat: {lat:.2f}, Lon: {lon:.2f}",
        labels={"year": "Year", var: "Value"},
        color_discrete_sequence=[color]
    )
    
    # Add trend line
    fig.add_scatter(
        x=ts_data["year"],
        y=ts_data["trend"],
        mode="lines",
        name="Trend Line",
        line=dict(color="red", dash="dash")
    )
    
    # Annotate slope, p-value, and trend direction in top-right corner
    annotation_text = f"Slope: {slope:.4f}<br>p-value: {p_value:.4f}<br>Trend: {trend_direction}"
    x_pos = ts_data["year"].max()  # Right edge of x-axis
    y_pos = ts_data[var].min()  # High, but not max, value of y

    fig.add_annotation(
        x=x_pos,
        y=y_pos,
        text=annotation_text,
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bgcolor="white",
        bordercolor="black",
        borderpad=4,
        opacity=0.8,
        font=dict(size=12)
    )    
    # Update layout
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=var,
        height=600,
        width=800,
        showlegend=True
    )
    
    return fig

# ------------------------------------
# Section C: Solara App
# ------------------------------------
@solara.component
def Layout(children=[]):
    return solara.AppLayout(children=children, sidebar_open=False)

@solara.component
def Page():
    # Reactive state for error messages, figure, DataFrame, click data
    error, set_error = solara.use_state(None)
    fig, set_fig = solara.use_state(None)
    df_state, set_df_state = solara.use_state(None)  # Store DataFrame
    click_data, set_click_data = solara.use_state(None)
    ts_fig, set_ts_fig = solara.use_state(None)  # Store time series figure
    
    # Reactive states for sidebar options
    selected_file_id, set_selected_file_id = solara.use_state(DEFAULT_FILE_ID)
    value_column, set_value_column = solara.use_state(None)
    color_scale, set_color_scale = solara.use_state(None)
    map_style, set_map_style = solara.use_state(DEFAULT_MAP_STYLE)
    opacity, set_opacity = solara.use_state(DEFAULT_OPACITY)
    available_columns, set_available_columns = solara.use_state([])
    
    # Reactive states for latitude and longitude inputs
    latitude, set_latitude = solara.use_state(0.0)
    longitude, set_longitude = solara.use_state(0.0)

    # Custom click handler to process click data and generate time series plot
    def handle_click_event(data):
        set_click_data(data)
        # Update latitude and longitude inputs from click data and generate plot
        if data and "points" in data and "point_indexes" in data["points"]:
            point_indexes = data["points"]["point_indexes"]
            if point_indexes and df_state is not None:
                try:
                    idx = point_indexes[0]
                    if idx < len(df_state):
                        selected_row = df_state.iloc[idx]
                        lat = float(selected_row["lat"])
                        lon = float(selected_row["lon"])
                        set_latitude(lat)
                        set_longitude(lon)
                        # Generate time series plot immediately
                        selected_file = next((f for f in FILE_CONFIG if f["id"] == selected_file_id), FILE_CONFIG[0])
                        ts_file_path = selected_file["ts"]
                        ts_plot = read_time_series(ts_file_path, lat, lon, DEFAULT_TS_COLOR)
                        set_ts_fig(ts_plot)
                        set_error(None)
                except Exception as e:
                    set_error(f"Error processing click data: {e}")
                    set_ts_fig(None)
    
    # Handle generate plot button click
    def generate_time_series_plot():
        try:
            selected_file = next((f for f in FILE_CONFIG if f["id"] == selected_file_id), FILE_CONFIG[0])
            ts_file_path = selected_file["ts"]
            ts_plot = read_time_series(ts_file_path, latitude, longitude, DEFAULT_TS_COLOR)
            set_ts_fig(ts_plot)
            set_error(None)
        except Exception as e:
            set_error(f"Error generating time series plot: {e}")
            set_ts_fig(None)

    # Load data and create plot
    def load_and_plot():
        try:
            # Find the selected file config
            selected_file = next((f for f in FILE_CONFIG if f["id"] == selected_file_id), FILE_CONFIG[0])
            df, file_path, float_vars, cell_dimensions = load_netcdf_data(selected_file, selected_file["no_data"])
            
            # Update available columns
            set_available_columns([col for col in float_vars if col not in ["lat", "lon", "latitude", "longitude"]])
            
            # Set default value column if not set or invalid
            if not value_column or value_column not in float_vars:
                set_value_column(float_vars[0] if float_vars else None)
            
            # Filter DataFrame to include only the selected value column, lat, and lon
            if value_column:
                dfics = df[value_column].notnull()
                columns_to_keep = ["lat", "lon", value_column]
                df = df[dfics][columns_to_keep]
                
                file_name = selected_file["name"]
                heatmap_fig = plot_heatmap(df, file_name, value_column, cell_dimensions, color_scale, map_style, opacity)
                set_fig(heatmap_fig)
                set_df_state(df)  # Store DataFrame in reactive state
                set_error(None)
        except Exception as e:
            set_error(f"Error loading data: {e}")
            set_fig(None)
            set_df_state(None)
    
    # Run the loading and plotting logic whenever dependencies change
    solara.use_effect(load_and_plot, dependencies=[selected_file_id, value_column, color_scale, map_style, opacity])
    
    # Update color scale when file changes
    def update_color_scale():
        selected_file = next((f for f in FILE_CONFIG if f["id"] == selected_file_id), FILE_CONFIG[0])
        set_color_scale(selected_file["default_color_scale"])
    
    solara.use_effect(update_color_scale, dependencies=[selected_file_id])
    
    # Render the UI
    with solara.AppBar():
        solara.Image("data/logo/TERI 50 Year Logo Seal.png", width="80px", classes=["mx-2"])
        solara.AppBarTitle("Rainfall Data Visualization")
        solara.lab.ThemeToggle()  # Theme toggle button

    with solara.Sidebar():
        with solara.Card("Map Controls", margin=0, elevation=0):
            with solara.Column():
                solara.Markdown("### Customize Map Appearance")
                
                solara.Select(
                    label="Select Dataset",
                    value=selected_file_id,
                    values=[f["id"] for f in FILE_CONFIG],
                    dense=True,
                    on_value=lambda id: set_selected_file_id(id)
                )
                
                solara.Select(
                    label="Value Column",
                    value=value_column,
                    values=available_columns,
                    on_value=set_value_column,
                    disabled=not available_columns
                )
                
                solara.Select(
                    label="Color Scale",
                    value=color_scale,
                    values=COLOR_SCALE_OPTIONS,
                    on_value=set_color_scale
                )
                solara.Select(
                    label="Map Style",
                   #value=map -_style,
                    value=map_style,
                    values=MAP_STYLE_OPTIONS,
                    on_value=set_map_style
                )
                solara.SliderFloat(
                    label="Opacity",
                    value=opacity,
                    min=0.1,
                    max=1.0,
                    step=0.1,
                    on_value=set_opacity
                )
        
        with solara.Card("Time Series Settings", margin=0, elevation=0):
            with solara.Column():
                solara.Markdown("### Coordinates")
                solara.InputFloat(
                    label="Latitude",
                    value=latitude,
                    on_value=set_latitude
                )
                solara.InputFloat(
                    label="Longitude",
                    value=longitude,
                    on_value=set_longitude
                )
                solara.Button(
                    label="Generate Plot",
                    color="primary",
                    text=True,
                    on_click=generate_time_series_plot
                )

    with solara.VBox():
        if error:
            solara.Error(error)
        elif fig:
            with solara.HBox():
                with solara.Column(classes=["w-3/4"]):
                    solara.FigurePlotly(fig, on_click=handle_click_event)
                
                with solara.Column(classes=["w-1/4"]):
                    solara.Markdown("## Time Series Plot")
                    if ts_fig:
                        solara.FigurePlotly(ts_fig)
                    else:
                        solara.Info("Click a point on the map or generate a time series plot using the button in the sidebar.")
        else:
            solara.Info("Loading data...")