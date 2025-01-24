import argparse
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
import os
import numpy as np

def load_data() -> pd.DataFrame:
    """Load the meat consumption dataset"""
    print("Downloading dataset...")
    meat_consumption_path = kagglehub.dataset_download("scibearia/meat-consumption-per-capita")
    
    try:
        df = pd.read_csv(f"{meat_consumption_path}/Consumption of meat per capita.csv")
        # Fix column names by stripping whitespace
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        print("Error: Could not find the expected CSV file in the downloaded dataset")
        exit(1)

def process_data(df: pd.DataFrame, country: str) -> Tuple[pd.DataFrame, List[str]]:
    """Process data for selected country and calculate 'other' meat type"""
    # Filter for selected country data
    country_data = df[df['Entity'] == country]
    
    if country_data.empty:
        print(f"Error: No data found for country '{country}'")
        exit(1)
        
    # Define meat types and years
    meat_types = ['Poultry', 'Beef', 'Sheep and goat', 'Pork']
    years = range(1961, 2022)
    
    # Filter years and calculate 'other' meat type
    country_data = country_data[(country_data['Year'] >= 1961) & 
                               (country_data['Year'] <= 2021)].copy()
    
    # Validate required columns
    required_columns = meat_types + ['Entity', 'Year']
    missing_columns = [col for col in required_columns if col not in country_data.columns]
    
    if missing_columns:
        print(f"Error: Dataset is missing required columns: {missing_columns}")
        print("Available columns:", list(country_data.columns))
        exit(1)
        
    # Calculate total meat if not present
    if 'Meat total' not in country_data.columns:
        print("Calculating total meat consumption from individual types...")
        country_data['Meat total'] = country_data[meat_types].sum(axis=1)
    
    return country_data, meat_types

def create_plot(country_data: pd.DataFrame, meat_types: List[str], country: str):
    """Create matplotlib visualization"""
    plt.figure(figsize=(12, 6))
    
    # Create stacked area plot
    plt.stackplot(country_data['Year'], 
                 [country_data[meat] for meat in meat_types],
                 labels=meat_types)
    
    plt.title(f'{country_data["Entity"].iloc[0]} Meat Consumption Breakdown (1961-2021)')
    plt.xlabel('Year')
    plt.ylabel('Meat Consumption per Capita (kg)')
    plt.legend(title='Meat Type')
    plt.grid(True)
    plt.xticks(range(1960, 2025, 5))
    
    # Save plot
    output_dir = "output/stacked-line-meat"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{country.lower().replace(' ', '_')}_meat_consumption.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to: {output_path}")

def main():
    # Load data first to get available countries
    df = load_data()
    
    # Get unique countries from dataset
    available_countries = df['Entity'].unique()
    
    # Get user input
    while True:
        country = input("Enter country name to analyze: ").strip()
        
        # Check if country exists in dataset
        if country in available_countries:
            break
        else:
            print(f"'{country}' not found in dataset. Available countries include:")
            print(", ".join(sorted(available_countries)))
            print("Please try again.\n")
    
    # Process data
    country_data, meat_types = process_data(df, country)
    
    # Create and save visualization
    create_plot(country_data, meat_types, country)

if __name__ == "__main__":
    main()
