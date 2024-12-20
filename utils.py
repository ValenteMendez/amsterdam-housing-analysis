import pandas as pd
import requests
import time
from typing import Tuple, Optional, List
import streamlit as st
from sklearn.ensemble import IsolationForest
import folium
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import json
from helpers import format_price  
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the CSV data."""
    df = pd.read_csv(file_path)
    
    # Convert numeric columns
    df['price_numeric'] = pd.to_numeric(df['price_numeric'], errors='coerce')
    df['living_area_numeric'] = pd.to_numeric(df['living_area_numeric'], errors='coerce')
    
    # Extract neighborhood from zip code
    df['neighborhood'] = df['zip'].str[:4]
    
    # Calculate price per square meter
    df['price_per_sqm'] = df['price_numeric'] / df['living_area_numeric']
    
    return df

def batch_geocode(addresses: List[Tuple[str, str]], batch_size: int = 10) -> List[Tuple[float, float]]:
    """Batch geocode addresses using Google Geocoding API."""
    api_key = os.getenv('GOOGLE_API_KEY')
    coordinates = []
    
    def geocode_single(address_data: Tuple[str, str]) -> Tuple[Optional[float], Optional[float]]:
        address, zip_code = address_data
        address_string = f"{address}, {zip_code}, Netherlands"
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={'+'.join(address_string.split())}&key={api_key}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    location = data['results'][0]['geometry']['location']
                    return location['lat'], location['lng']
            return None, None
        except Exception:
            return None, None

    # Process in batches using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        coordinates = list(executor.map(geocode_single, addresses))
        time.sleep(0.1)  # Small delay between batches
    
    return coordinates

def identify_undervalued_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Identify undervalued properties using local market comparisons"""
    # Create area categories
    df['area_category'] = pd.qcut(
        df['living_area_numeric'], 
        q=3, 
        labels=['small', 'medium', 'large']
    )
    
    # Extract neighborhood from postal code (first 4 digits)
    df['neighborhood'] = df['zip'].str[:4]
    
    # Calculate price per m² with validation
    df['price_per_sqm'] = df.apply(
        lambda x: x['price_numeric'] / x['living_area_numeric'] 
        if x['living_area_numeric'] > 0 and pd.notna(x['living_area_numeric'])
        else pd.NA, 
        axis=1
    )
    
    # Group by similar characteristics
    df['is_undervalued'] = False
    
    for neighborhood in df['neighborhood'].unique():
        for house_type in df['house_type'].unique():
            for area_cat in ['small', 'medium', 'large']:
                # Filter similar properties
                mask = (
                    (df['neighborhood'] == neighborhood) &
                    (df['house_type'] == house_type) &
                    (df['area_category'] == area_cat) &
                    df['price_per_sqm'].notna()
                )
                
                similar_properties = df[mask].copy()
                
                if len(similar_properties) > 10:  # Minimum sample size
                    # Calculate local metrics
                    local_median = similar_properties['price_per_sqm'].median()
                    
                    # Isolation Forest on local group
                    model = IsolationForest(
                        contamination=0.1, 
                        random_state=42
                    )
                    
                    similar_properties['anomaly'] = model.fit_predict(
                        similar_properties[['price_per_sqm']]
                    )
                    
                    # Mark as undervalued if anomaly and below local median
                    similar_properties['is_undervalued'] = (
                        (similar_properties['anomaly'] == -1) &
                        (similar_properties['price_per_sqm'] < local_median)
                    )
                    
                    # Update main dataframe
                    df.loc[mask, 'is_undervalued'] = similar_properties['is_undervalued']
    
    return df

def create_property_map(df: pd.DataFrame) -> folium.Map:
    """Create a map with property markers and layer controls."""
    m = folium.Map(
        location=[52.3676, 4.9041],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    folium.TileLayer(
        'CartoDB positron',
        name='CartoDB'
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    for _, row in df.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            color = 'red' if row['is_undervalued'] else 'blue'
            
            # Handle NaN in living area
            living_area = (
                f"{int(row['living_area_numeric'])} m²" 
                if pd.notna(row['living_area_numeric']) 
                else "N/A"
            )
            
            popup_text = (
                f"Price: {format_price(row['price_numeric'])}<br>"
                f"Area: {living_area}<br>"
                f"ID: <a href='{row['url']}' target='_blank'>{row['house_id']}</a>"
            )
            
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=200),
                icon=folium.Icon(color=color, icon='home', prefix='fa')
            ).add_to(m)
    
    return m

def analyze_market(df: pd.DataFrame):
    """Perform market analysis on property data."""
    # Price prediction
    features = df[['living_area_numeric']].fillna(df['living_area_numeric'].mean())
    target = df['price_numeric']
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    df['predicted_price'] = model.predict(features)
    
    # Clustering
    scaler = StandardScaler()
    cluster_features = scaler.fit_transform(
        df[['price_numeric', 'living_area_numeric']].fillna(0)
    )
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['market_segment'] = kmeans.fit_predict(cluster_features)
    
    return df

def is_valid_postal_code(postal_code: str) -> bool:
    """Validate Dutch postal code format (1234 AB)"""
    import re
    pattern = r'^\d{4}\s?[A-Z]{2}$'
    return bool(re.match(pattern, str(postal_code)))

def create_market_analysis_charts(df: pd.DataFrame, selected_neighborhoods=None):
    """Create market analysis charts with neighborhood filtering"""
    charts = {}
    
    # 1. Area vs Price Scatter (filtered)
    valid_data = df[
        (df['living_area_numeric'] > 0) & 
        (df['price_numeric'] > 0)
    ]
    
    charts['area_price'] = px.scatter(
        valid_data,
        x='living_area_numeric',
        y='price_numeric',
        title='living area vs price',
        labels={
            'living_area_numeric': 'living area (m²)',
            'price_numeric': 'price (€)'
        },
        height=600
    )
    
    # 2. Price by Postal Code (sorted)
    valid_postal_mask = (
        df['zip']
        .fillna('')
        .str.match(r'^\d{4}\s?[A-Z]{2}$')
        .fillna(False)
    )
    
    avg_price_by_zip = (df[valid_postal_mask]
                       .groupby('zip')['price_numeric']
                       .agg(['mean', 'count'])
                       .reset_index()
                       .sort_values('mean', ascending=False))
    
    charts['price_by_zip'] = px.bar(
        avg_price_by_zip,
        x='zip',
        y='mean',
        title='average price by postal code',
        labels={'mean': 'average price (€)', 'zip': 'postal code'},
        height=600
    ).update_layout(
        xaxis={'tickangle': 45},
        showlegend=False,
        margin=dict(b=100)
    )
    
    # 3. Full Price Distribution
    charts['price_dist'] = px.histogram(
        df,
        x='price_numeric',
        title='price distribution (all properties)',
        labels={'price_numeric': 'price (€)', 'count': 'number of properties'},
        height=500
    )
    
    # 4. Price Distribution under 1M
    charts['price_dist_1m'] = px.histogram(
        df[df['price_numeric'] <= 1_000_000],
        x='price_numeric',
        title='price distribution (properties under €1M)',
        labels={'price_numeric': 'price (€)', 'count': 'number of properties'},
        height=500
    )
    
    # 5. Price per Square Meter
    charts['price_per_sqm'] = px.histogram(
        df.dropna(subset=['price_per_sqm']),
        x='price_per_sqm',
        title='price per m² distribution',
        labels={
            'price_per_sqm': 'price per m² (€)',
            'count': 'number of properties'
        },
        height=500
    )
    
    # 6. Property Type Distribution
    charts['property_type'] = px.pie(
        df['house_type'].value_counts().reset_index(),
        values='count',
        names='house_type',
        title='distribution of property types',
        height=400
    )
    
    # 7. Energy Label Distribution
    charts['energy_label'] = px.bar(
        df['energy_label'].value_counts().reset_index(),
        x='energy_label',
        y='count',
        title='distribution of energy labels',
        labels={'energy_label': 'energy label', 'count': 'number of properties'},
        height=400
    )
    
    # 8. Room Count Distribution
    charts['room_dist'] = px.histogram(
        df,
        x='room',
        title='distribution of room counts',
        labels={'room': 'number of rooms', 'count': 'number of properties'},
        height=400
    )
    
    # 9. Price per m² by Property Type
    charts['price_by_type'] = px.box(
        df,
        x='house_type',
        y='price_per_sqm',
        title='price per m² by property type',
        labels={
            'house_type': 'property type',
            'price_per_sqm': 'price per m² (€)'
        },
        height=500
    )
    
    # 10. Room Count vs Price
    charts['rooms_price'] = px.scatter(
        df,
        x='room',
        y='price_numeric',
        title='room count vs price',
        labels={
            'room': 'number of rooms',
            'price_numeric': 'price (€)'
        },
        height=500
    )
    
    # Filter data for selected neighborhoods
    if selected_neighborhoods:
        neighborhood_data = df[df['neighborhood'].isin(selected_neighborhoods)]
    else:
        neighborhood_data = df
    
    # Create neighborhood comparison chart
    charts['neighborhood_price_sqm'] = px.box(
        neighborhood_data,
        x='neighborhood',
        y='price_per_sqm',
        title='price per m² distribution by selected neighborhoods',
        labels={
            'neighborhood': 'neighborhood',
            'price_per_sqm': 'price per m² (€)'
        },
        height=500
    ).update_layout(
        xaxis={'tickangle': 45},
        showlegend=False,
        margin=dict(b=100)
    )
    
    # Prepare data for treemaps by filtering NaN values
    treemap_data = df.dropna(subset=['neighborhood', 'house_type', 'price_numeric'])
    area_treemap_data = df.dropna(subset=['zip', 'house_type', 'living_area_numeric'])
    
    # Add Treemap visualization
    charts['price_tree'] = px.treemap(
        treemap_data,
        path=[px.Constant("Amsterdam"), 'zip', 'house_type'],  # Changed from neighborhood to zip
        values='price_numeric',
        title='price distribution by postal code and property type',
        height=600
    )
    
    # Add Area Treemap
    charts['area_tree'] = px.treemap(
        area_treemap_data,
        path=[px.Constant("Amsterdam"), 'zip', 'house_type'],
        values='living_area_numeric',
        title='living area distribution by postal code',
        height=600
    )
    
    # Add Violin plots with box plots
    charts['price_violin'] = px.violin(
        df,
        x='house_type',
        y='price_numeric',
        box=True,
        points="all",
        title='price distribution by property type',
        labels={
            'house_type': 'property type',
            'price_numeric': 'price (€)'
        },
        height=500
    )
    
    # Add neighborhood price distribution
    charts['neighborhood_box'] = px.box(
        df,
        x='neighborhood',
        y='price_per_sqm',
        points="all",
        title='price per m² distribution by neighborhood',
        labels={
            'neighborhood': 'neighborhood',
            'price_per_sqm': 'price per m² (€)'
        },
        height=500
    ).update_layout(xaxis={'tickangle': 45})
    
    return charts

def analyze_postal_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze average prices by postal code."""
    # Filter valid postal codes using regex pattern
    valid_postal_mask = df['zip'].str.match(r'^\d{4}\s?[A-Z]{2}$', na=False)
    valid_df = df[valid_postal_mask].copy()
    
    # Calculate average price per postal code
    postal_analysis = (valid_df.groupby('zip')
                      .agg({
                          'price_numeric': ['mean', 'count'],
                          'price_per_sqm': 'mean'
                      })
                      .round(2)
                      .reset_index())
    
    # Flatten column names
    postal_analysis.columns = ['postal_code', 'avg_price', 'num_properties', 'avg_price_sqm']
    
    # Sort by average price descending
    postal_analysis = postal_analysis.sort_values('avg_price', ascending=False)
    
    return postal_analysis