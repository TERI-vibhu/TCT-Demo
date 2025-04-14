import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------------- Data Handling Functions ----------------------

def load_default_data(file_path):
    try:
        df = pd.read_csv(file_path)
        st.sidebar.success("Using default dataset")
        return df
    except FileNotFoundError:
        st.sidebar.error(f"Default file not found: {file_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading default data: {str(e)}")
    return None

def load_uploaded_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Successfully loaded: {uploaded_file.name}")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded file: {str(e)}")
    return None

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

# ---------------------- Visualization Functions ----------------------

def create_grid_heatmap(df, x_axis, y_axis, values, color_scale, map_style, zoom, center_lat, center_lon, cell_width, cell_height, grid_opacity):
    features = []
    for idx, row in df.iterrows():
        lon = row[x_axis]
        lat = row[y_axis]
        val = row[values]

        feature = {
            "type": "Feature",
            "id": idx,
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

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=[f["id"] for f in features],
        z=[f["properties"]["value"] for f in features],
        colorscale=color_scale,
        marker_opacity=grid_opacity,
        marker_line_width=0,
        colorbar=dict(title=values),
        customdata=np.stack((
            [f["properties"]["lat"] for f in features],
            [f["properties"]["lon"] for f in features]
        ), axis=-1),
        hovertemplate="<b>Lat: %{customdata[0]:.3f}</b><br>Lon: %{customdata[1]:.3f}<br>Value: %{z}<extra></extra>"
    ))

    fig.update_layout(
        mapbox=dict(style=map_style, center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        height=1000,
        width=900
    )

    return fig

def create_scatter_plot(df, x_axis, y_axis, values, color_scale, map_style, zoom, center_lat, center_lon, point_size, point_opacity):
    fig = go.Figure(go.Scattermapbox(
        lat=df[y_axis],
        lon=df[x_axis],
        mode='markers',
        marker=dict(
            size=point_size,
            color=df[values],
            colorscale=color_scale,
            opacity=point_opacity,
            colorbar=dict(title=values),
        ),
        hovertemplate="<b>Lat: %{lat:.3f}</b><br>Lon: %{lon:.3f}<br>Value: %{marker.color}<extra></extra>"
    ))

    fig.update_layout(
        mapbox=dict(style=map_style, center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        height=1200,
        width=1000
    )

    return fig

# ---------------------- Streamlit UI ----------------------

st.title("TCT Demo")
st.sidebar.header("Controls")

default_file_path = "/home/vibhu/Downloads/TCT/gridpoint_temperature_stats_max.csv"

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use default dataset", "Upload my own CSV"],
    help="Select whether to use the built-in dataset or upload your own CSV file"
)

df = None
if data_source == "Use default dataset":
    df = load_default_data(default_file_path)
else:
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv", 
                                             help="Upload your own CSV file with 'lon' and 'lat' columns")
    if uploaded_file:
        df = load_uploaded_data(uploaded_file)

if df is not None:
    show_raw_data = st.sidebar.checkbox("Show raw data", value=False,
                                        help="Display the raw data table below the visualization")
    if show_raw_data:
        st.subheader("Raw data")
        st.write(df)

    viz_type = st.sidebar.radio("Select visualization type:", ["Grid Heatmap", "Scatter Plot"],
                                help="Choose between grid cells (heatmap) or points (scatter)")

    x_axis, y_axis = 'lon', 'lat'

    if x_axis in df.columns and y_axis in df.columns:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        value_options = [col for col in numeric_cols if col not in [x_axis, y_axis]]

        if value_options:
            values = st.sidebar.selectbox("Select values column", value_options,
                                          help="Choose which data column to visualize with colors")
            map_style = st.sidebar.selectbox("Select map style", [
                "open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner"
            ], help="Change the background map appearance")
            color_scale = st.sidebar.selectbox("Color scale", [
                'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Reds'
            ], help="Select the color palette for visualization")

            center_lat = df[y_axis].mean()
            center_lon = df[x_axis].mean()
            zoom = int(calculate_zoom(np.ptp(df[y_axis].values), np.ptp(df[x_axis].values)))

            if viz_type == "Grid Heatmap":
                x_vals = df[x_axis].unique()
                y_vals = df[y_axis].unique()
                if len(x_vals) > 1:
                    x_diff = np.min(np.diff(np.sort(x_vals)))
                else:
                    x_diff = 0.1
                if len(y_vals) > 1:
                    y_diff = np.min(np.diff(np.sort(y_vals)))
                else:
                    y_diff = 0.1

                with st.sidebar.expander("Advanced Settings", expanded=False):
                    cell_width = st.number_input("Cell Width (longitude)", value=float(x_diff), step=0.001, format="%.5f",
                                                 help="Width of each grid cell in longitude units")
                    cell_height = st.number_input("Cell Height (latitude)", value=float(y_diff), step=0.001, format="%.5f",
                                                  help="Height of each grid cell in latitude units")
                    grid_opacity = st.number_input("Grid opacity", min_value=0.0, max_value=1.0, value=0.8, step=0.05,
                                                   help="Transparency of grid cells (0=transparent, 1=opaque)")

                fig = create_grid_heatmap(df, x_axis, y_axis, values, color_scale, map_style, zoom,
                                          center_lat, center_lon, cell_width, cell_height, grid_opacity)
                st.plotly_chart(fig, use_container_width=True)

            else:  # Scatter Plot
                with st.sidebar.expander("Advanced Settings", expanded=False):
                    point_size = st.number_input("Point Size", min_value=2, max_value=20, value=6, step=1,
                                                 help="Size of each data point on the map")
                    point_opacity = st.number_input("Point Opacity", min_value=0.0, max_value=1.0, value=0.8, step=0.05,
                                                    help="Transparency of points (0=transparent, 1=opaque)")

                fig = create_scatter_plot(df, x_axis, y_axis, values, color_scale, map_style, zoom,
                                          center_lat, center_lon, point_size, point_opacity)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No suitable numerical columns found for values")
    else:
        missing_cols = []
        if x_axis not in df.columns:
            missing_cols.append(x_axis)
        if y_axis not in df.columns:
            missing_cols.append(y_axis)
        st.error(f"Required column(s) missing from the dataset: {', '.join(missing_cols)}")
        st.info("Your CSV file must contain 'lon' and 'lat' columns for this visualization.")
else:
    if data_source == "Use default dataset":
        st.info("Default dataset could not be loaded. Please check the file path or upload your own file.")
    else:
        st.info("Please upload a CSV file to get started.")
