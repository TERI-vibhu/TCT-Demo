import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xarray as xr
import solara
from uuid import uuid4
from sklearn.neighbors import NearestNeighbors
import solara.lab  # Import for ThemeToggle

# ------------------------------------
# Section A: Configuration
# ------------------------------------
FILE_CONFIG = {
    "name": "Rainfall",
    "description": "Rainfall data from IMD",
    "ts": "data/RF_yearly.nc",  # Time series NetCDF file
    "avg": "data/IMD_RF_AVG_after1994.nc",
    "no_data": -999
}
COLOR_SCALE = "Blues"
MAP_STYLE = "carto-positron"
GRID_OPACITY = 0.8

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
    
    selected_var = float_vars[0]
    if no_data_value is not None:
        ds[selected_var] = ds[selected_var].where(ds[selected_var] != no_data_value, np.nan)
    
    df = ds[selected_var].to_dataframe().reset_index().dropna()
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
        print(f"Average distance to nearest neighbor (grid spacing estimate): {avg_distance:.5f} degrees")
    else:
        print("Not enough data points to estimate grid spacing.")
    
    return df, file_path, selected_var, avg_distance

def calculate_zoom(lat_range, lon_range):
    r = max(lat_range, lon_range)
    return 1 if r > 100 else 2 if r > 30 else 4 if r > 15 else 5 if r > 5 else 7 if r > 1 else 10 if r > 0.1 else 13

def plot_heatmap(df, file_name, value_col, cell_dimensions):
    CELL_HEIGHT = cell_dimensions
    CELL_WIDTH = cell_dimensions
    """Creates a heatmap using Plotly Choroplethmapbox with a map overlay."""
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
        go.Choroplethmapbox(
            geojson=geojson,
            locations=[f["id"] for f in features],
            z=[f["properties"]["value"] for f in features],
            colorscale=COLOR_SCALE,
            marker_opacity=GRID_OPACITY,
            marker_line_width=0,
            colorbar=dict(title=value_col),
            customdata=np.stack(([f["properties"]["lat"] for f in features], [f["properties"]["lon"] for f in features]), axis=-1),
            hovertemplate="<b>Lat: %{customdata[0]:.5f}</b><br>Lon: %{customdata[1]:.5f}<br>Value: %{z}<extra></extra>"
        )
    )
    fig.update_layout(
        mapbox=dict(style=MAP_STYLE, center=center, zoom=zoom),
        height=900,
        width=1000,
        title=f"Heatmap - {file_name}"
    )
    return fig

# ------------------------------------
# Section C: Solara App
# ------------------------------------
@solara.component
def Page():
    # Reactive state for error messages, figure, DataFrame, and click data
    error, set_error = solara.use_state(None)
    fig, set_fig = solara.use_state(None)
    df_state, set_df_state = solara.use_state(None)  # Store DataFrame
    click_data, set_click_data = solara.use_state(None)
    
    # Custom click handler to process click data
    def handle_click_event(data):
        print("Click event detected:", data)
        set_click_data(data)
    
    # Load data and create plot on component mount
    def load_and_plot():
        try:
            df, file_path, value_col, cell_dimensions = load_netcdf_data(FILE_CONFIG, FILE_CONFIG["no_data"])
            if value_col != "value":
                df = df.rename(columns={value_col: "value"})
                value_col = "value"
            
            file_name = FILE_CONFIG["name"]
            heatmap_fig = plot_heatmap(df, file_name, value_col, cell_dimensions)
            set_fig(heatmap_fig)
            set_df_state(df)  # Store DataFrame in reactive state
            set_error(None)
        except Exception as e:
            set_error(f"Error loading data: {e}")
            set_fig(None)
            set_df_state(None)
    
    # Run the loading and plotting logic once on mount
    solara.use_effect(load_and_plot, dependencies=[])
    
    # Render the UI
    with solara.VBox():
        # Add AppBarTitle with text, image, and theme toggle
        with solara.AppBarTitle():
            with solara.HBox(align_items="center"):
                solara.Image("data/TERI Logo Seal.png", width="50px", classes=["mx-2"])
                solara.Text("Rainfall Data Visualization")
                solara.lab.ThemeToggle()  # Add theme toggle button
                
        if error:
            solara.Error(error)
        elif fig:
            with solara.HBox():
                with solara.Column(classes=["w-3/4"]):
                    solara.FigurePlotly(fig, on_click=handle_click_event)
                
                with solara.Column(classes=["w-1/4"]):
                    solara.Markdown("## Click Data")
                    if click_data:
                        # Extract point_indexes
                        point_indexes = click_data.get("points", {}).get("point_indexes", [])
                        solara.Markdown(f"**Point Indexes**: {point_indexes}")
                        
                        # Extract the row from the DataFrame if point_indexes is valid
                        if point_indexes and df_state is not None:
                            try:
                                # Get the first index from point_indexes (e.g., [9] -> 9)
                                idx = point_indexes[0]
                                if idx < len(df_state):
                                    selected_row = df_state.iloc[idx]
                                    solara.Markdown("**Selected Row from DataFrame**:")
                                    # Convert row to a markdown table for display
                                    row_df = pd.DataFrame([selected_row])
                                    solara.Markdown(row_df.to_markdown(index=False))
                                else:
                                    solara.Warning(f"Index {idx} is out of bounds for DataFrame with {len(df_state)} rows.")
                            except Exception as e:
                                solara.Error(f"Error extracting row: {e}")
                        else:
                            solara.Info("No valid point index or DataFrame available.")
                    else:
                        solara.Info("Click on the map to see data")
        else:
            solara.Info("Loading data...")