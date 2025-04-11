import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

# Set up the Streamlit app
st.title("TCT Demo")

# Configure the sidebar for controls
st.sidebar.header("Controls")

# Default file path - replace with your actual file path
default_file_path = "gridpoint_temperature_stats_max.csv"  # Change this to your actual default file path

# Let users choose between default file or uploading their own
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use default dataset", "Upload my own CSV"],
    help="Select whether to use the built-in dataset or upload your own CSV file"
)

df = None

if data_source == "Use default dataset":
    try:
        # Read the predefined CSV data
        df = pd.read_csv(default_file_path)
        st.sidebar.success(f"Using default dataset")
    except FileNotFoundError:
        st.sidebar.error(f"Default file not found: {default_file_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading default data: {str(e)}")
else:
    # Allow user to upload a CSV file
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload your own CSV file with 'lon' and 'lat' columns"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Successfully loaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading uploaded file: {str(e)}")

if df is not None:
    # Display the raw data (optional)
    show_raw_data = st.sidebar.checkbox(
        "Show raw data", 
        value=False,
        help="Display the raw data table below the visualization"
    )
    
    if show_raw_data:
        st.subheader("Raw data")
        st.write(df)
    
    # Add visualization type selector
    viz_type = st.sidebar.radio(
        "Select visualization type:",
        ["Grid Heatmap", "Scatter Plot"],
        help="Choose between grid cells (heatmap) or points (scatter)"
    )
    
    # Fixed X and Y axes to 'lon' and 'lat'
    x_axis = 'lon'
    y_axis = 'lat'
    
    # Check if the required columns exist
    if x_axis in df.columns and y_axis in df.columns:
        # Create a copy of the dataframe to avoid modifying the original
        plot_df = df.copy()
        
        # Only show numerical columns for the values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            # Exclude coordinate columns from the value options
            value_options = [col for col in numeric_cols if col not in [x_axis, y_axis]]
            if value_options:
                values = st.sidebar.selectbox(
                    "Select values column", 
                    value_options,
                    help="Choose which data column to visualize with colors"
                )
                
                # Select map style
                map_style = st.sidebar.selectbox(
                    "Select map style", 
                    [
                        "open-street-map", 
                        "carto-positron", 
                        "carto-darkmatter", 
                        "stamen-terrain", 
                        "stamen-toner"
                    ],
                    help="Change the background map appearance"
                )
                 # Number input for opacity
                grid_opacity = 0.8 
                grid_opacity = st.sidebar.number_input(
                            "Grid opacity", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=grid_opacity, 
                            step=0.05,
                            help="Transparency of grid cells (0=transparent, 1=opaque)"
                        )

                
                # Calculate center of the map
                center_lat = plot_df[y_axis].mean()
                center_lon = plot_df[x_axis].mean()
                
                # Determine appropriate zoom level based on data spread
                lat_range = plot_df[y_axis].max() - plot_df[y_axis].min()
                lon_range = plot_df[x_axis].max() - plot_df[x_axis].min()
                max_range = max(lat_range, lon_range)
                
                # Simple heuristic for zoom level
                if max_range > 10:
                    zoom = 3.5
                elif max_range > 5:
                    zoom = 4
                elif max_range > 1:
                    zoom = 6
                elif max_range > 0.1:
                    zoom = 8
                else:
                    zoom = 11
                
                # Fixed zoom level (removed slider)
                zoom = int(zoom)
                
                # Select color scale
                color_scale = st.sidebar.selectbox(
                    "Color scale", 
                    ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Reds'],
                    help="Select the color palette for visualization"
                )
                
                if viz_type == "Grid Heatmap":
                    # Calculate default cell size based on data spread
                    x_vals = plot_df[x_axis].unique()
                    y_vals = plot_df[y_axis].unique()
                    
                    if len(x_vals) > 1 and len(y_vals) > 1:
                        x_sorted = np.sort(x_vals)
                        y_sorted = np.sort(y_vals)
                        x_diff = np.min(np.diff(x_sorted))
                        y_diff = np.min(np.diff(y_sorted))
                    else:
                        # Default values if we can't calculate
                        x_diff = 0.1
                        y_diff = 0.1
                    
                    # Default opacity value before advanced settings
                    
                    cell_width = float(x_diff)
                    cell_height = float(y_diff)
                    
                    # Put Grid Cell Settings in a collapsible expander labeled "Advanced Settings"
                    with st.sidebar.expander("Advanced Settings", expanded=False):
                        st.write("Grid Cell Settings")
                        
                        cell_width = st.number_input(
                            "Cell Width (longitude)", 
                            value=cell_width, 
                            step=0.001, 
                            format="%.5f",
                            help="Width of each grid cell in longitude units"
                        )
                        
                        cell_height = st.number_input(
                            "Cell Height (latitude)", 
                            value=cell_height, 
                            step=0.001, 
                            format="%.5f",
                            help="Height of each grid cell in latitude units"
                        )
                        
                                           
                    # Create GeoJSON-like features for each grid cell
                    features = []
                    
                    # Store the location information for hover data
                    location_info = {}
                    
                    for _, row in plot_df.iterrows():
                        # Create a square feature for this grid cell
                        lon = row[x_axis]
                        lat = row[y_axis]
                        val = row[values]
                        
                        # Generate an ID for this feature
                        feature_id = len(features)
                        
                        # Store the location info for hover
                        location_info[feature_id] = {
                            "lat": lat,
                            "lon": lon,
                            "value": val
                        }
                        
                        # Calculate corners of the grid cell
                        half_width = cell_width / 2
                        half_height = cell_height / 2
                        
                        # GeoJSON geometry for a polygon (square in this case)
                        feature = {
                            "type": "Feature",
                            "id": feature_id,  # Unique ID for each feature
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [lon - half_width, lat - half_height],  # bottom-left
                                    [lon + half_width, lat - half_height],  # bottom-right
                                    [lon + half_width, lat + half_height],  # top-right
                                    [lon - half_width, lat + half_height],  # top-left
                                    [lon - half_width, lat - half_height]   # close the polygon
                                ]]
                            },
                            "properties": {
                                "value": val,
                                "lat": lat,
                                "lon": lon
                            }
                        }
                        features.append(feature)
                    
                    # Create GeoJSON structure
                    geojson = {
                        "type": "FeatureCollection",
                        "features": features
                    }
                    
                    # Create a mapping between feature ID and value for choropleth
                    locations = [f["id"] for f in features]
                    color_values = [f["properties"]["value"] for f in features]
                    
                    # Also create arrays for hover information
                    hover_lats = [f["properties"]["lat"] for f in features]
                    hover_lons = [f["properties"]["lon"] for f in features]
                    
                    # Create the choropleth mapbox with custom data for hover
                    fig = go.Figure(go.Choroplethmapbox(
                        geojson=geojson,
                        locations=locations,
                        z=color_values,
                        colorscale=color_scale,
                        marker_opacity=grid_opacity,
                        marker_line_width=0,
                        colorbar=dict(title=values),
                        customdata=np.stack((hover_lats, hover_lons), axis=-1),
                        hovertemplate="<b>Lat: %{customdata[0]:.5f}</b><br>Lon: %{customdata[1]:.5f}<br>Value: %{z}<extra></extra>"
                    ))
                    
                    # Update the layout for mapbox
                    fig.update_layout(
                        mapbox=dict(
                            style=map_style,
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=zoom
                        ),
                        height=700,
                        width=1000
                    )
                
                else:  # Scatter Plot
                    # Default scatter plot values
                    point_size = 6
                    point_opacity = 0.8
                    
                    # Put Scatter Plot Settings in a collapsible expander labeled "Advanced Settings"
                    with st.sidebar.expander("Advanced Settings", expanded=False):
                        st.write("Scatter Plot Settings")
                        
                        # Add point size control for scatter plot
                        point_size = st.number_input(
                            "Point Size", 
                            min_value=2, 
                            max_value=20, 
                            value=point_size, 
                            step=1,
                            help="Size of each data point on the map"
                        )
                        
                        point_opacity = st.number_input(
                            "Point Opacity", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=point_opacity, 
                            step=0.05,
                            help="Transparency of points (0=transparent, 1=opaque)"
                        )
                    
                    # Create the scatter plot on a map
                    fig = go.Figure(go.Scattermapbox(
                        lat=plot_df[y_axis],
                        lon=plot_df[x_axis],
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            color=plot_df[values],
                            colorscale=color_scale,
                            opacity=point_opacity,
                            colorbar=dict(title=values),
                        ),
                        hovertemplate="<b>Lat: %{lat:.5f}</b><br>Lon: %{lon:.5f}<br>Value: %{marker.color}<extra></extra>"
                    ))
                    
                    # Update the layout for mapbox
                    fig.update_layout(
                        mapbox=dict(
                            style=map_style,
                            center=dict(lat=center_lat, lon=center_lon),
                            zoom=zoom
                        ),
                        height=700,
                        width=1000
                    )
                
                # Display the figure
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No suitable numerical columns found for values")
        else:
            st.error("No numerical columns found in the data for creating a visualization")
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
        st.info("Default dataset could not be loaded. Please check the file path or try uploading your own file.")
    else:
        st.info("Please upload a CSV file to get started")
