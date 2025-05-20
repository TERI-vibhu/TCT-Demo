
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xarray as xr
import solara
from uuid import uuid4
from sklearn.neighbors import NearestNeighbors
import solara.lab
from scipy.stats import linregress
import geopandas as gpd
from shapely.geometry import Point
import netCDF4 as nc

# ------------------------------------
# Section A: Configuration
# ------------------------------------
DEFAULT_DATASET = "Rainfall (IMD)"
DEFAULT_VIS_MODE = "Heatmap"
SHAPEFILE_PATH = "data/boundry/India_State_Boundary.shp"
REGION_COLUMN = "Name"

DATASETS = [
    {
        "id": "Rainfall (IMD)",
        "name": "Rainfall Annual (mm)",
        "description": "IMD Rainfall 1985-2014",
        "timeseries_path": "data/Processed Datasets/IMD_RF_Yearly-Mean.nc",
        "average_path": "data/Processed Datasets/IMD_Rainfall_1985-2014_Climatology.nc",
        "no_data": -999.0,
        "color_scale": "Blues"
    },
    {
        "id": "Tmax (IMD)",
        "name": "Tmax Annual (°C)",
        "description": "IMD Maximum Temperature 1985-2014",
        "timeseries_path": "data/Processed Datasets/IMD_MaxT_Yearly-Mean_1951-2023.nc",
        "average_path": "data/Processed Datasets/IMD_MaxT_1985-2014_Climatology.nc",
        "no_data": 99.9,
        "color_scale": "Reds"
    },
    {
        "id": "Tmin (IMD)",
        "name": "Tmin Annual °C",
        "description": "IMD Minimum Temperature 1985-2014",
        "timeseries_path": "data/Processed Datasets/IMD_MinT_Yearly-Mean_1951-2023.nc",
        "average_path": "data/Processed Datasets/IMD_MinT_1985-2014_Climatology.nc",
        "no_data": 99.9,
        "color_scale": "Reds"
    }
]

COLOR_SCALES = ["Blues", "Viridis", "Reds", "Greens", "Inferno", "Magma"]
MAP_STYLES = ["carto-positron", "carto-darkmatter", "open-street-map", "stamen-terrain"]
DEFAULT_MAP_STYLE = "carto-darkmatter"
DEFAULT_OPACITY = 0.8
TIMESERIES_COLOR = "cyan"

# ------------------------------------
# Section B: Utility Functions
# ------------------------------------
def get_plotly_theme():
    return "plotly_dark" if solara.lab.theme.dark_effective else "plotly_white"

def load_dataset(dataset_config):
    dataset = xr.open_dataset(dataset_config["average_path"])
    variables = list(dataset.data_vars)
    float_vars = [var for var in variables if dataset[var].dtype in ["float32", "float64"]]
    
    data = dataset.to_dataframe().reset_index().dropna()
    data = data.rename(columns={"latitude": "lat", "longitude": "lon"})
    
    if len(data) >= 4:
        sample_indices = np.random.choice(len(data), size=4, replace=False)
        sample_coords = data[["lat", "lon"]].iloc[sample_indices].values
        coords = data[["lat", "lon"]].values
        
        knn = NearestNeighbors(n_neighbors=2).fit(coords)
        distances, _ = knn.kneighbors(sample_coords, n_neighbors=2)
        grid_spacing = distances[:, 1].mean()
    else:
        grid_spacing = 0.25
    
    dataset.close()
    return data, float_vars, grid_spacing

def calculate_map_zoom(lat_range, lon_range):
    range_max = max(lat_range, lon_range)
    return (1 if range_max > 100 else 
            2 if range_max > 30 else 
            4 if range_max > 15 else 
            5 if range_max > 5 else 
            7 if range_max > 1 else 
            9 if range_max > 0.1 else 10)


def create_heatmap(data, dataset_name, value_column, grid_size, color_scale, map_style, opacity):
    center = {"lat": data["lat"].mean(), "lon": data["lon"].mean()}
    zoom = calculate_map_zoom(data["lat"].max() - data["lat"].min(), 
                            data["lon"].max() - data["lon"].min())
    
    features = [
        {
            "type": "Feature", "id": str(uuid4()),
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [row["lon"] - grid_size/2, row["lat"] - grid_size/2],
                    [row["lon"] + grid_size/2, row["lat"] - grid_size/2],
                    [row["lon"] + grid_size/2, row["lat"] + grid_size/2],
                    [row["lon"] - grid_size/2, row["lat"] + grid_size/2],
                    [row["lon"] - grid_size/2, row["lat"] - grid_size/2]
                ]]
            },
            "properties": {"value": row[value_column], "lat": row["lat"], "lon": row["lon"]}
        }
        for _, row in data.iterrows()
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
            colorbar=dict(title=dataset_name),  # Use dataset_name for colorbar title
            customdata=np.stack(([f["properties"]["lat"] for f in features], 
                               [f["properties"]["lon"] for f in features]), axis=-1),
            hovertemplate="<b>Lat: %{customdata[0]:.3f}</b><br>Lon: %{customdata[1]:.3f}<br>Value: %{z}<extra></extra>"
        )
    )
    fig.update_layout(
        map=dict(style=map_style, center=center, zoom=zoom),
        height=900, 
        width=1000,
        title=f"Heatmap - {dataset_name}",
        template=get_plotly_theme()
    )
    return fig



def create_timeseries_plot(file_path, dataset_name, lat, lon, color, spatial_mean=None, show_spatial_mean=False, from_year=None, to_year=None, use_timeframe=False):
    try:
        dataset = xr.open_dataset(file_path)
        variable = list(dataset.data_vars)[0]
        
        lat_idx = np.abs((dataset.get("lat", dataset.get("latitude")).values - lat)).argmin()
        lon_idx = np.abs((dataset.get("lon", dataset.get("longitude")).values - lon)).argmin()
        
        timeseries = dataset[variable].isel(lat=lat_idx, lon=lon_idx).to_dataframe().reset_index()
        
        # Extract start and end years from the dataset
        years = timeseries["year"].values
        if len(years) == 0:
            raise ValueError("No years found in the dataset")
        start_year = int(min(years))
        end_year = int(max(years))
        
        # Filter by time frame if use_timeframe is True and from_year/to_year are provided
        if use_timeframe and from_year is not None and to_year is not None:
            timeseries = timeseries[(timeseries["year"] >= from_year) & (timeseries["year"] <= to_year)]
            start_year = from_year
            end_year = to_year
        
        x = timeseries["year"].values
        y = timeseries[variable].values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        trend = "Increasing" if p_value < 0.05 and slope > 0 else "Decreasing" if p_value < 0.05 else "No Significant Trend"
        trend_line = intercept + slope * x
        timeseries["trend"] = trend_line
        
        # Debug print to verify title components
        print(f"Creating timeseries plot: dataset_name={dataset_name}, start_year={start_year}, end_year={end_year}, lat={lat:.2f}, lon={lon:.2f}")
        
        fig = px.line(
            timeseries, 
            x="year", 
            y=variable,
            title=f"{dataset_name}, {start_year}-{end_year} for Lat: {lat:.2f}, Lon: {lon:.2f}",
            labels={"year": "Year", variable: dataset_name},
            color_discrete_sequence=[color],
            render_mode="svg"
        )
        fig.data[0].name = dataset_name
        
        fig.add_scatter(
            x=timeseries["year"], 
            y=timeseries["trend"],
            mode="lines", 
            name=f"{dataset_name} Trend",
            line=dict(color="red", dash="dash")
        )
        
        if show_spatial_mean and spatial_mean is not None:
            # Add spatial mean as a scatter trace, repeating the mean value across all years
            spatial_mean_line = np.full_like(x, spatial_mean)
            fig.add_scatter(
                x=timeseries["year"],
                y=spatial_mean_line,
                mode="lines",
                name="Spatial Mean",
                line=dict(color="yellow", dash="dot", width=2)
            )
        
        fig.add_annotation(
            x=timeseries["year"].max(),
            y=timeseries[variable].min(),
            text=f"Slope: {slope:.4f}<br>p-value: {p_value:.4f}<br>Trend: {trend}",
            showarrow=False, 
            xanchor="right", 
            yanchor="top",
            bgcolor="white" if not solara.lab.theme.dark_effective else "#333333",
            bordercolor="black", 
            borderpad=4, 
            opacity=0.8,
            font=dict(size=12, color="black" if not solara.lab.theme.dark_effective else "white")
        )
        
        fig.update_layout(
            xaxis_title="Year", 
            yaxis_title=dataset_name,
            height=600, 
            width=800, 
            showlegend=True,
            legend=dict(
                x=0.99,  # Position legend inside, near top-right
                y=0.99,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(0, 0, 0, 0)"  # Fully transparent background
            ),
            template=get_plotly_theme()
        )
        
        dataset.close()
        return fig
    
    except Exception as e:
        print(f"Error creating timeseries plot: {e}")
        dataset.close() if 'dataset' in locals() else None
        return go.Figure()
def create_3d_surface(file_path, dataset_name, value_column, color_scale):
    try:
        dataset = nc.Dataset(file_path, mode='r')
        
        lons = dataset.variables['lon'][:] if 'lon' in dataset.variables else dataset.variables['longitude'][:]
        lats = dataset.variables['lat'][:] if 'lat' in dataset.variables else dataset.variables['latitude'][:]
        data = np.squeeze(dataset.variables[value_column][:])
        
        fig = go.Figure(
            go.Surface(
                z=data,
                x=lons,
                y=lats,
                colorscale=color_scale,
                colorbar=dict(title=value_column),
                lighting=dict(ambient=0.8, diffuse=0.8),
                showscale=True
            )
        )
        
        fig.update_layout(
            title=dict(
                text=f"3D Surface - {dataset_name}",
                x=0.5,
                xanchor="center",
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                zaxis_title=value_column,
                xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.5)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.5)"),
                zaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.5)"),
                bgcolor="rgba(0,0,0,0)"
            ),
            height=900,
            width=1000,
            margin=dict(l=20, r=20, t=60, b=20),
            template=get_plotly_theme()
        )
        
        dataset.close()
        return fig
    
    except Exception as e:
        print(f"Error creating 3D surface plot: {e}")
        dataset.close()
        return go.Figure()

def get_region_bounds(geodataframe, region):
    geometry = geodataframe[geodataframe[REGION_COLUMN] == region].geometry.iloc[0]
    minx, miny, maxx, maxy = geometry.bounds
    return {"min_lon": minx, "max_lon": maxx, "min_lat": miny, "max_lat": maxy}

def filter_region_data(data, geodataframe, region):
    if not region or region == "India":
        return data
        
    points = gpd.GeoDataFrame(
        data, geometry=gpd.points_from_xy(data["lon"], data["lat"]), crs=geodataframe.crs
    )
    region_geometry = geodataframe[geodataframe[REGION_COLUMN] == region].geometry.iloc[0]
    filtered_points = points[points.geometry.within(region_geometry)]
    return filtered_points.drop(columns=["geometry"])

def calculate_spatial_mean(dataset_config, geodataframe, region, from_year=None, to_year=None, use_timeframe=False):
    try:
        dataset = xr.open_dataset(dataset_config["timeseries_path"])
        variable = list(dataset.data_vars)[0]
        no_data_value = dataset_config["no_data"]

        data = dataset[variable].to_dataframe().reset_index()
        data = data.rename(columns={"latitude": "lat", "longitude": "lon"})

        data = data[data[variable] != no_data_value]

        if use_timeframe and from_year is not None and to_year is not None:
            data = data[(data["year"] >= from_year) & (data["year"] <= to_year)]

        if region != "India" and geodataframe is not None:
            data = filter_region_data(data, geodataframe, region)
        
        spatial_mean = data[variable].mean()
        
        dataset.close()
        return spatial_mean
    except Exception as e:
        print(f"Error calculating spatial mean: {e}")
        return None

def calculate_selected_spatial_mean(dataset_config, selected_coords, from_year=None, to_year=None, use_timeframe=False):
    try:
        if not selected_coords:
            return None
        
        dataset = xr.open_dataset(dataset_config["timeseries_path"])
        variable = list(dataset.data_vars)[0]
        no_data_value = dataset_config["no_data"]

        data_list = []
        for lat, lon in selected_coords:
            lat_idx = np.abs((dataset.get("lat", dataset.get("latitude")).values - lat)).argmin()
            lon_idx = np.abs((dataset.get("lon", dataset.get("longitude")).values - lon)).argmin()
            point_data = dataset[variable].isel(lat=lat_idx, lon=lon_idx).to_dataframe().reset_index()
            data_list.append(point_data)
        
        data = pd.concat(data_list, ignore_index=True)
        data = data[data[variable] != no_data_value]

        if use_timeframe and from_year is not None and to_year is not None:
            data = data[(data["year"] >= from_year) & (data["year"] <= to_year)]

        spatial_mean = data[variable].mean()
        
        dataset.close()
        return spatial_mean
    except Exception as e:
        print(f"Error calculating selected spatial mean: {e}")
        return None

def create_selected_timeseries_plot(file_path, dataset_name, selected_coords, color, selected_spatial_mean=None, from_year=None, to_year=None, use_timeframe=False):
    try:
        if not selected_coords:
            return go.Figure()
        
        dataset = xr.open_dataset(file_path)
        variable = list(dataset.data_vars)[0]
        no_data_value = next(d for d in DATASETS if d["name"] == dataset_name)["no_data"]

        data_list = []
        for lat, lon in selected_coords:
            lat_idx = np.abs((dataset.get("lat", dataset.get("latitude")).values - lat)).argmin()
            lon_idx = np.abs((dataset.get("lon", dataset.get("longitude")).values - lon)).argmin()
            point_data = dataset[variable].isel(lat=lat_idx, lon=lon_idx).to_dataframe().reset_index()
            data_list.append(point_data)
        
        data = pd.concat(data_list, ignore_index=True)
        data = data[data[variable] != no_data_value]

        # Group by year and calculate mean
        timeseries = data.groupby("year")[variable].mean().reset_index()

        # Extract start and end years
        years = timeseries["year"].values
        if len(years) == 0:
            raise ValueError("No years found in the dataset")
        start_year = int(min(years))
        end_year = int(max(years))

        # Filter by time frame if use_timeframe is True
        if use_timeframe and from_year is not None and to_year is not None:
            timeseries = timeseries[(timeseries["year"] >= from_year) & (timeseries["year"] <= to_year)]
            start_year = from_year
            end_year = to_year

        x = timeseries["year"].values
        y = timeseries[variable].values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        trend = "Increasing" if p_value < 0.05 and slope > 0 else "Decreasing" if p_value < 0.05 else "No Significant Trend"
        trend_line = intercept + slope * x
        timeseries["trend"] = trend_line
        
        fig = px.line(
            timeseries, 
            x="year", 
            y=variable,
            title=f"{dataset_name}, Spatial Mean of Selected Points, {start_year}-{end_year}",
            labels={"year": "Year", variable: dataset_name},
            color_discrete_sequence=[color]
        )
        fig.data[0].name = dataset_name
        
        fig.add_scatter(
            x=timeseries["year"], 
            y=timeseries["trend"],
            mode="lines", 
            name=f"{dataset_name} Trend",
            line=dict(color="red", dash="dash")
        )
        
        if selected_spatial_mean is not None:
            # Add spatial mean as a scatter trace, repeating the mean value across all years
            spatial_mean_line = np.full_like(x, selected_spatial_mean)
            fig.add_scatter(
                x=timeseries["year"],
                y=spatial_mean_line,
                mode="lines",
                name="Spatial Mean",
                line=dict(color="yellow", dash="dot", width=2)
            )
        
        fig.add_annotation(
            x=timeseries["year"].max(),
            y=timeseries[variable].min(),
            text=f"Slope: {slope:.4f}<br>p-value: {p_value:.4f}<br>Trend: {trend}",
            showarrow=False, 
            xanchor="right", 
            yanchor="top",
            bgcolor="white" if not solara.lab.theme.dark_effective else "#333333",
            bordercolor="black", 
            borderpad=4, 
            opacity=0.8,
            font=dict(size=12, color="black" if not solara.lab.theme.dark_effective else "white")
        )
        
        fig.update_layout(
            xaxis_title="Year", 
            yaxis_title=f"Mean {dataset_name}",
            height=600, 
            width=800, 
            showlegend=True,
            legend=dict(
                x=0.99,  # Position legend inside, near top-right
                y=0.99,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(0, 0, 0, 0)"  # Fully transparent background
            ),
            template=get_plotly_theme()
        )
        
        dataset.close()
        return fig
    
    except Exception as e:
        print(f"Error creating selected timeseries plot: {e}")
        dataset.close() if 'dataset' in locals() else None
        return go.Figure()
def get_available_years(dataset_config):
    try:
        dataset = xr.open_dataset(dataset_config["timeseries_path"])
        years = dataset["year"].values.tolist()
        dataset.close()
        return sorted(years)
    except Exception as e:
        print(f"Error getting available years: {e}")
        return []

# ------------------------------------
# Section C: Solara App
# ------------------------------------
tab_index = solara.reactive(0)  # Main tabs: Heatmap, 3D Plot
right_tab_index = solara.reactive(0)  # Right column tabs: Time Series Plot, Selected Points Analysis

@solara.component
def Layout(children=[]):
    return solara.AppLayout(children=children, sidebar_open=False)

@solara.component
def Page():
    dataset_id, set_dataset_id = solara.use_state(DEFAULT_DATASET)
    data, set_data = solara.use_state(None)
    heatmap, set_heatmap = solara.use_state(None)
    timeseries, set_timeseries = solara.use_state(None)
    selected_timeseries, set_selected_timeseries = solara.use_state(None)
    surface_3d, set_surface_3d = solara.use_state(None)
    value_column, set_value_column = solara.use_state(None)
    color_scale, set_color_scale = solara.use_state(None)
    map_style, set_map_style = solara.use_state(DEFAULT_MAP_STYLE)
    opacity, set_opacity = solara.use_state(DEFAULT_OPACITY)
    variables, set_variables = solara.use_state([])
    latitude, set_latitude = solara.use_state(0.0)
    longitude, set_longitude = solara.use_state(0.0)
    vis_mode, set_vis_mode = solara.use_state(DEFAULT_VIS_MODE)
    use_regions, set_use_regions = solara.use_state(False)
    region, set_region = solara.use_state("India")
    regions, set_regions = solara.use_state(["India"])
    geodataframe, set_geodataframe = solara.use_state(None)
    region_bounds, set_region_bounds = solara.use_state(None)
    show_spatial_mean, set_show_spatial_mean = solara.use_state(False)
    spatial_mean, set_spatial_mean = solara.use_state(None)
    selected_spatial_mean, set_selected_spatial_mean = solara.use_state(None)
    available_years, set_available_years = solara.use_state([])
    from_year, set_from_year = solara.use_state(None)
    to_year, set_to_year = solara.use_state(None)
    use_timeframe, set_use_timeframe = solara.use_state(False)
    selected_coords, set_selected_coords = solara.use_state([])
    show_stats_for_nerds, set_show_stats_for_nerds = solara.use_state(False)
    show_timeseries_plot, set_show_timeseries_plot = solara.use_state(True)
    show_selected_timeseries_plot, set_show_selected_timeseries_plot = solara.use_state(True)

    # Map tab_index to vis_mode
    def sync_tab_with_vis_mode():
        if tab_index.value == 0 and vis_mode != "Heatmap":
            set_vis_mode("Heatmap")
        elif tab_index.value == 1 and vis_mode != "3D Plot":
            set_vis_mode("3D Plot")

    # Update tab_index when vis_mode changes
    def sync_vis_mode_with_tab():
        if vis_mode == "Heatmap" and tab_index.value != 0:
            tab_index.set(0)
        elif vis_mode == "3D Plot" and tab_index.value != 1:
            tab_index.set(1)

    solara.use_effect(sync_tab_with_vis_mode, dependencies=[tab_index])
    solara.use_effect(sync_vis_mode_with_tab, dependencies=[vis_mode])

    def load_shapefile():
        try:
            gdf = gpd.read_file(SHAPEFILE_PATH)
            set_regions(["India"] + sorted(gdf[REGION_COLUMN].unique().tolist()))
            set_geodataframe(gdf)
        except Exception as e:
            print(f"Error loading shapefile: {e}")

    solara.use_effect(load_shapefile, dependencies=[])

    def update_region_bounds():
        if geodataframe is not None and region != "India" and use_regions:
            try:
                set_region_bounds(get_region_bounds(geodataframe, region))
            except Exception as e:
                print(f"Error getting region bounds: {e}")
                set_region_bounds(None)
        else:
            set_region_bounds(None)

    solara.use_effect(update_region_bounds, dependencies=[region, use_regions, geodataframe])

    def load_available_years():
        dataset = next(d for d in DATASETS if d["id"] == dataset_id)
        years = get_available_years(dataset)
        set_available_years(years)
        if years:
            set_from_year(years[0])
            set_to_year(years[-1])
        else:
            set_from_year(None)
            set_to_year(None)

    solara.use_effect(load_available_years, dependencies=[dataset_id])

    def compute_spatial_mean():
        if not show_spatial_mean:
            set_spatial_mean(None)
            return
        dataset = next(d for d in DATASETS if d["id"] == dataset_id)
        mean_value = calculate_spatial_mean(dataset, geodataframe, region, from_year, to_year, use_timeframe)
        set_spatial_mean(mean_value)

    solara.use_effect(compute_spatial_mean, dependencies=[dataset_id, region, use_regions, geodataframe, show_spatial_mean, from_year, to_year, use_timeframe])

    def compute_selected_spatial_mean():
        if not selected_coords:
            set_selected_spatial_mean(None)
            set_selected_timeseries(None)
            return
        dataset = next(d for d in DATASETS if d["id"] == dataset_id)
        mean_value = calculate_selected_spatial_mean(dataset, selected_coords, from_year, to_year, use_timeframe)
        set_selected_spatial_mean(mean_value)
        set_selected_timeseries(create_selected_timeseries_plot(
            dataset["timeseries_path"],
            dataset["name"],
            selected_coords,
            TIMESERIES_COLOR,
            mean_value,
            from_year,
            to_year,
            use_timeframe
        ))

    solara.use_effect(compute_selected_spatial_mean, dependencies=[dataset_id, selected_coords, from_year, to_year, use_timeframe])

    def handle_map_click(click_data):
        if click_data and "points" in click_data and "point_indexes" in click_data["points"]:
            idx = click_data["points"]["point_indexes"][0]
            row = data.iloc[idx]
            set_latitude(float(row["lat"]))
            set_longitude(float(row["lon"]))
            dataset = next(d for d in DATASETS if d["id"] == dataset_id)
            set_timeseries(create_timeseries_plot(
                dataset["timeseries_path"], 
                dataset["name"],
                float(row["lat"]), 
                float(row["lon"]), 
                TIMESERIES_COLOR,
                spatial_mean=spatial_mean,
                show_spatial_mean=show_spatial_mean,
                from_year=from_year,
                to_year=to_year,
                use_timeframe=use_timeframe
            ))
            if not show_selected_timeseries_plot:
                set_show_selected_timeseries_plot(False)

    def handle_map_selection(selection_data):
        if selection_data and "points" in selection_data and "point_indexes" in selection_data["points"]:
            selected_indices = selection_data["points"]["point_indexes"]
            coords = [(float(data.iloc[idx]["lat"]), float(data.iloc[idx]["lon"])) for idx in selected_indices]
            set_selected_coords(coords)
            if not show_timeseries_plot:
                set_show_timeseries_plot(False)
        else:
            set_selected_coords([])

    def generate_timeseries():
        dataset = next(d for d in DATASETS if d["id"] == dataset_id)
        set_timeseries(create_timeseries_plot(
            dataset["timeseries_path"], 
            dataset["name"],
            latitude, 
            longitude, 
            TIMESERIES_COLOR,
            spatial_mean=spatial_mean,
            show_spatial_mean=show_spatial_mean,
            from_year=from_year,
            to_year=to_year,
            use_timeframe=use_timeframe
        ))
        if not show_selected_timeseries_plot:
            set_show_selected_timeseries_plot(False)

    def load_and_visualize():
        dataset = next(d for d in DATASETS if d["id"] == dataset_id)
        loaded_data, float_vars, grid_size = load_dataset(dataset)
        
        set_variables([var for var in float_vars if var not in ["lat", "lon", "latitude", "longitude"]])
        new_value_column = float_vars[0] if float_vars else None
        set_value_column(new_value_column)
        
        if use_regions and geodataframe is not None:
            loaded_data = filter_region_data(loaded_data, geodataframe, region)
        
        if not new_value_column:
            return
        
        filtered_data = loaded_data[loaded_data[new_value_column].notnull()][["lat", "lon", new_value_column]]
        
        set_heatmap(create_heatmap(filtered_data, dataset["name"], new_value_column, 
                                 grid_size, color_scale, map_style, opacity))
        set_surface_3d(create_3d_surface(dataset["average_path"], dataset["name"], 
                                       new_value_column, color_scale))
        set_data(filtered_data)
        set_selected_coords([])

    solara.use_effect(load_and_visualize, 
                     dependencies=[dataset_id, color_scale, map_style, 
                                  opacity, region, use_regions])

    def update_color_scale():
        dataset = next(d for d in DATASETS if d["id"] == dataset_id)
        set_color_scale(dataset["color_scale"])

    solara.use_effect(update_color_scale, dependencies=[dataset_id])

    def sync_map_style_with_theme():
        if map_style not in ["carto-darkmatter", "carto-positron"]:
            return
        new_map_style = "carto-darkmatter" if solara.lab.theme.dark_effective else "carto-positron"
        set_map_style(new_map_style)

    solara.use_effect(sync_map_style_with_theme, dependencies=[solara.lab.theme.dark_effective])

    with solara.AppBar():
        solara.Image("data/logo/TERI 50 Year Logo Seal.png", width="80px", classes=["mx-2"])
        solara.AppBarTitle("TERI Climate Data Explorer")
        solara.lab.ThemeToggle()

    with solara.Sidebar():
        with solara.Card("Controls", margin=0, elevation=0):
            with solara.Column():
                solara.Details(
                    summary="Dataset Settings",
                    children=[
                        solara.Select(
                            label="Dataset",
                            value=dataset_id,
                            values=[d["id"] for d in DATASETS],
                            dense=True,
                            on_value=set_dataset_id
                        ),
                        solara.Select(
                            label="Value",
                            value=value_column,
                            values=variables,
                            on_value=set_value_column,
                            disabled=not variables
                        )
                    ],
                    expand=False
                )
                solara.Details(
                    summary="Visualization Settings",
                    children=[
                        solara.Select(
                            label="Color Scale",
                            value=color_scale,
                            values=COLOR_SCALES,
                            on_value=set_color_scale
                        ),
                        solara.Select(
                            label="Map Style",
                            value=map_style,
                            values=MAP_STYLES,
                            on_value=set_map_style
                        ),
                        solara.SliderFloat(
                            label="Opacity",
                            value=opacity,
                            min=0.1, max=1.0, step=0.1,
                            on_value=set_opacity
                        )
                    ],
                    expand=False
                )
                solara.Details(
                    summary="Time Series Settings",
                    children=[
                        solara.Markdown("### Coordinates"),
                        solara.InputFloat(label="Latitude", value=latitude, on_value=set_latitude),
                        solara.InputFloat(label="Longitude", value=longitude, on_value=set_longitude),
                        solara.Checkbox(
                            label="Use Specific Time Frame",
                            value=use_timeframe,
                            on_value=set_use_timeframe
                        ),
                        solara.Select(
                            label="From Year",
                            value=from_year,
                            values=available_years,
                            on_value=set_from_year,
                            disabled=not available_years or not use_timeframe
                        ),
                        solara.Select(
                            label="To Year",
                            value=to_year,
                            values=[y for y in available_years if y >= (from_year or available_years[0])],
                            on_value=set_to_year,
                            disabled=not available_years or not use_timeframe
                        ),
                        solara.Button(label="Generate Plot", color="primary", 
                                    text=True, on_click=generate_timeseries)
                    ],
                    expand=False
                )
                solara.Details(
                    summary="Plot Visibility",
                    children=[
                        solara.Checkbox(
                            label="Show Time Series Plot",
                            value=show_timeseries_plot,
                            on_value=set_show_timeseries_plot
                        ),
                        solara.Checkbox(
                            label="Show Selected Points Timeseries Plot",
                            value=show_selected_timeseries_plot,
                            on_value=set_show_selected_timeseries_plot
                        )
                    ],
                    expand=False
                )

        with solara.Card("Regionalization", margin=0, elevation=0):
            with solara.Column():
                solara.Details(
                    summary="Region Selection",
                    children=[
                        solara.Select(
                            label="Region",
                            value=region,
                            values=regions,
                            on_value=lambda value: [set_region(value), set_use_regions(value != "India")]
                        ),
                        solara.Checkbox(
                            label="Show Spatial Mean",
                            value=show_spatial_mean,
                            on_value=set_show_spatial_mean
                        ),
                        solara.Markdown("### Spatial Mean Value"),
                        solara.Markdown(f"**Spatial Mean ({from_year}-{to_year}):** {spatial_mean:.2f}" if show_spatial_mean and spatial_mean is not None and use_timeframe and from_year and to_year else 
                                        f"**Spatial Mean (All Years):** {spatial_mean:.2f}" if show_spatial_mean and spatial_mean is not None else 
                                        "**Spatial Mean:** Not available" if show_spatial_mean else "")
                    ],
                    expand=False
                )

    with solara.VBox():
        with solara.lab.Tabs(value=tab_index):
            with solara.lab.Tab("Heatmap"):
                if vis_mode == "Heatmap" and heatmap:
                    with solara.HBox():
                        with solara.Column(classes=["w-3/4"]):
                            solara.FigurePlotly(heatmap, on_click=handle_map_click, on_selection=handle_map_selection)
                        
                        with solara.Column(classes=["w-1/4"]):
                            with solara.lab.Tabs(value=right_tab_index):
                                with solara.lab.Tab("Time Series Plot"):
                                    if timeseries and show_timeseries_plot:
                                        solara.FigurePlotly(timeseries)
                                    elif not show_timeseries_plot and timeseries:
                                        solara.Info("Time Series Plot is disabled. Enable it in the sidebar.")
                                    else:
                                        solara.Info("Click a map point or generate a time series plot.")
                                with solara.lab.Tab("Selected Points Analysis"):
                                    if selected_coords:
                                        solara.Markdown(f"**Spatial Mean of Selected Points{' (' + str(from_year) + '-' + str(to_year) + ')' if use_timeframe and from_year and to_year else ''}:** {selected_spatial_mean:.2f}" if selected_spatial_mean is not None else "**Spatial Mean of Selected Points:** Not available")
                                        if selected_timeseries and show_selected_timeseries_plot:
                                            solara.FigurePlotly(selected_timeseries)
                                        elif not show_selected_timeseries_plot and selected_timeseries:
                                            solara.Info("Selected Points Timeseries Plot is disabled. Enable it in the sidebar.")
                                        else:
                                            solara.Info("Select a region on the heatmap to view spatial mean and timeseries.")
                                    else:
                                        solara.Info("Select a region on the heatmap to view spatial mean and timeseries.")
                                    solara.Checkbox(
                                        label="Stats for Nerds (Show Selected Coordinates)",
                                        value=show_stats_for_nerds,
                                        on_value=set_show_stats_for_nerds
                                    )
                                    if show_stats_for_nerds and selected_coords:
                                        coords_df = pd.DataFrame(selected_coords, columns=["Latitude", "Longitude"])
                                        solara.Markdown(f"**Total Number of Selected Points:** {len(selected_coords)}")
                                        solara.DataFrame(coords_df, items_per_page=5)
                                    elif show_stats_for_nerds:
                                        solara.Info("Select a region on the heatmap to view coordinates.")
            with solara.lab.Tab("3D Plot"):
                if vis_mode == "3D Plot" and surface_3d:
                    solara.FigurePlotly(surface_3d)
                else:
                    solara.Info("Loading 3D plot...")