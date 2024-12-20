import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import time
import os
import json
import plotly.express as px  # Add this import
from utils import (
    load_data,
    batch_geocode,
    identify_undervalued_properties,
    create_property_map,
    analyze_market,  # Add this import
    create_market_analysis_charts  # Add this import
)
from helpers import format_price

# Configuration
st.set_page_config(
    page_title="Funda Property Analysis",
    page_icon=":house:",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Amsterdam Property Analysis - December 2024")
    
    # File paths using relative path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "funda_listings_with_coords.csv")
    
    # Load and process data
    try:
        df = load_data(file_path)
    except FileNotFoundError:
        st.error(f"Error: Could not find the file at {file_path}")
        return
    
    # Add coordinates if not already present
    if 'latitude' not in df.columns:
        try:
            # Try to load from cache first
            with open(coords_cache_path, 'r') as f:
                coords_cache = json.load(f)
            df['latitude'] = [coord[0] for coord in coords_cache]
            df['longitude'] = [coord[1] for coord in coords_cache]
        except (FileNotFoundError, json.JSONDecodeError):
            with st.spinner("Adding geolocation data..."):
                # Create address tuples for geocoding
                addresses = list(zip(df['address'], df['zip']))
                # Batch geocode addresses
                coordinates = batch_geocode(addresses)
                # Save coordinates to cache
                with open(coords_cache_path, 'w') as f:
                    json.dump(coordinates, f)
                # Add coordinates to dataframe
                df['latitude'] = [coord[0] for coord in coordinates]
                df['longitude'] = [coord[1] for coord in coordinates]
    
    # Identify undervalued properties
    df = identify_undervalued_properties(df)
    
    # Calculate price range values
    min_price = int(df['price_numeric'].min())
    max_price = int(df['price_numeric'].max())
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["map view", "market analysis"])
    
    with tab1:
        # Map filters section
        st.subheader("Map filter")
        col1, col2 = st.columns(2)
        
        with col1:
            # Add unique key to checkbox
            show_undervalued = st.checkbox(
                "show only undervalued properties",
                key="map_undervalued_filter"
            )
        
        with col2:
            # Create custom price range options
            price_steps = [
                50_000, 75_000, 100_000, 125_000, 150_000, 175_000, 200_000,
                225_000, 250_000, 275_000, 300_000, 325_000, 350_000, 375_000, 400_000,
                450_000, 500_000, 550_000, 600_000, 650_000, 700_000, 750_000,
                800_000, 900_000, 1_000_000, 1_250_000, 1_500_000, 2_000_000,
                2_500_000, 3_000_000, 3_500_000, 4_000_000, 4_500_000, 5_000_000
            ]
            price_options = [format_price(p) for p in price_steps]
            
            # Create columns for min/max selection
            price_col1, price_col2 = st.columns(2)
            
            with price_col1:
                min_price_selected = st.selectbox(
                    "min price",
                    options=price_options[:-1],
                    index=0
                )
            
            with price_col2:
                min_idx = price_options.index(min_price_selected)
                max_price_selected = st.selectbox(
                    "max price",
                    options=price_options[min_idx + 1:],
                    index=len(price_options[min_idx + 1:]) - 1
                )
            
            # Convert selected prices back to numeric
            min_price_value = float(min_price_selected.replace('€', '').replace(',', ''))
            max_price_value = float(max_price_selected.replace('€', '').replace(',', ''))
        
        # Update mask with new price range values
        mask = (
            (df['price_numeric'] >= min_price_value) &
            (df['price_numeric'] <= max_price_value)
        )
        if show_undervalued:
            mask &= df['is_undervalued']
        
        df_filtered = df[mask]
        
        # Map section with statistics
        map_col1, map_col2 = st.columns([3, 1])
        
        with map_col1:
            st.subheader("Map of properties for sale in Amsterdam")
            st.text("Remember to activate openstreetmap in the settings and click on a marker to see more details")
            property_map = create_property_map(df_filtered)
            st_folium(
                property_map,
                width="100%",
                height=600,
                returned_objects=["last_active_drawing"],
                use_container_width=True
            )
        
        with map_col2:
            st.subheader("High level stats")
            st.metric(
                "Total properties", 
                f"{len(df_filtered):,}"  # Add comma separator
            )
            st.metric(
                "Average price", 
                format_price(df_filtered['price_numeric'].mean())
            )
            st.metric(
                "Undervalued properties", 
                df_filtered['is_undervalued'].sum()
            )
            if 'price_per_sqm' in df_filtered.columns:
                st.metric(
                    "Average price/m²", 
                    format_price(df_filtered['price_per_sqm'].mean())
                )
        st.markdown("*Note: undervalued properties follow a proprietary algorithm, contact us for additional detail*")
    
    with tab2:
        
        # Add summary statistics section
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Listings",
                f"{len(df):,}"  # Add comma separator
            )
            st.metric(
                "Average Price",
                format_price(df['price_numeric'].mean())
            )
            st.metric(
                "Median Price",
                format_price(df['price_numeric'].median())
            )
            st.metric(
                "Total Market Value",
                format_price(df['price_numeric'].sum())
            )
        
        with col2:
            st.metric(
                "Average Living Area",
                f"{df['living_area_numeric'].mean():.1f} m²"
            )
            st.metric(
                "Median Living Area",
                f"{df['living_area_numeric'].median():.1f} m²"
            )
            st.metric(
                "Average Price/m²",
                format_price(df['price_per_sqm'].mean())
            )
            st.metric(
                "Total Living Area",
                f"{df['living_area_numeric'].sum():,.0f} m²"
            )
        
        with col3:
            st.metric(
                "Houses",
                f"{len(df[df['house_type'] == 'House']):,}"  # Add comma separator
            )
            st.metric(
                "Apartments", 
                f"{len(df[df['house_type'] == 'Apartment']):,}"  # Add comma separator
            )
            st.metric(
                "Postal Codes",
                f"{df['zip'].nunique():,}"  # Add comma separator
            )
        
        st.subheader("Market Insights")
        
        # Display unfiltered charts
        charts = create_market_analysis_charts(df)
        
        st.plotly_chart(charts['area_price'], use_container_width=True)
        st.plotly_chart(charts['price_by_zip'], use_container_width=True)
        st.plotly_chart(charts['price_dist'], use_container_width=True)
        st.plotly_chart(charts['price_dist_1m'], use_container_width=True)
        st.plotly_chart(charts['price_per_sqm'], use_container_width=True)
        st.plotly_chart(charts['property_type'], use_container_width=True)
        st.plotly_chart(charts['price_tree'], use_container_width=True)
        st.markdown(
            "*The tree map above shows the distribution of properties by price range. "
            "The size of each box represents the number of properties in that range.*"
        )
        st.plotly_chart(charts['area_tree'], use_container_width=True)
        st.markdown(
            "*The tree map above shows the distribution of properties by area range. "
            "The size of each box represents the number of properties in that range.*"
        )

if __name__ == "__main__":
    main()

# Footer
st.markdown(
    'Made by [Valentin Mendez](https://www.linkedin.com/in/valentemendez/) using information from [Funda NL](https://www.funda.nl/)'
)

# Hide the "Made with Streamlit" footer
hide_streamlit_style = """
<style>
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)