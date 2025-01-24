from flask import Flask, render_template, request
import pandas as pd
import plotly
import plotly.express as px
import json
import kagglehub
import os

app = Flask(__name__)

def load_data():
    """Load and cache the meat consumption dataset"""
    print("Downloading dataset...")
    meat_consumption_path = kagglehub.dataset_download("scibearia/meat-consumption-per-capita")
    
    try:
        return pd.read_csv(f"{meat_consumption_path}/Consumption of meat per capita.csv")
    except FileNotFoundError:
        return None

def process_data(df, country):
    """Process data for selected country and calculate 'other' meat type"""
    # Filter for selected country data
    country_data = df[df['Entity'] == country]
    
    # Define meat types and years
    meat_types = ['Poultry', 'Beef', 'Sheep and goat', 'Pork']
    years = range(1961, 2022)
    
    # Filter years and calculate 'other' meat type
    country_data = country_data[(country_data['Year'] >= 1961) & 
                               (country_data['Year'] <= 2021)].copy()
    country_data['Other'] = country_data['Meat total'] - country_data[meat_types].sum(axis=1)
    
    # Add 'Other' to meat types
    meat_types.append('Other')
    
    return country_data, meat_types

@app.route('/')
def index():
    df = load_data()
    if df is None:
        return "Error loading data", 500
    
    countries = df['Entity'].unique()
    default_country = 'China'
    
    # Get selected country from query parameters
    selected_country = request.args.get('country', default_country)
    
    # Process data for selected country
    country_data, meat_types = process_data(df, selected_country)
    
    # Create Plotly figure
    fig = px.area(country_data,
                 x='Year',
                 y=meat_types,
                 title=f'{selected_country} Meat Consumption Breakdown (1961-2021)',
                 labels={'value': 'Meat Consumption per Capita (kg)', 'variable': 'Meat Type'},
                 template='plotly_white')
    
    fig.update_layout(
        hovermode='x unified',
        legend_title='Meat Type',
        xaxis_title='Year',
        yaxis_title='Meat Consumption per Capita (kg)',
        xaxis=dict(tickmode='linear', dtick=5),
        height=600
    )
    
    # Convert plot to JSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('index.html', 
                         graphJSON=graphJSON,
                         countries=countries,
                         selected_country=selected_country)

if __name__ == '__main__':
    app.run(debug=True)
