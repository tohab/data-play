import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download meat consumption dataset
meat_consumption_path = kagglehub.dataset_download("vagifa/meatconsumption")
print("Meat consumption dataset path:", meat_consumption_path)

# Load meat consumption data
meat_df = pd.read_csv(f"{meat_consumption_path}/meat_consumption_worldwide.csv")

# Create country code to name mappings
meat_country_mapping = {
    'USA': 'United States',
    'CHN': 'China',
    'IND': 'India',
    'GBR': 'United Kingdom'
}

# Create reverse mapping for GDP dataset
gdp_country_mapping = {
    'Country_103': 'United States',
    'Country_15': 'China',
    'Country_93': 'India',
    'Country_107': 'United Kingdom'
}

# Filter for selected countries (US, China, India, UK) in meat data
countries = list(meat_country_mapping.keys())
filtered_meat = meat_df[meat_df['LOCATION'].isin(countries)]
print("Filtered meat data:\n", filtered_meat.head())

# Load GDP per capita data from global warming dataset
global_warming_path = kagglehub.dataset_download("ankushpanday1/global-warming-dataset-195-countries-1900-2023")
print("Global warming dataset path:", global_warming_path)

# List contents of downloaded dataset
import os
print("Dataset contents:", os.listdir(global_warming_path))

gdp_df = pd.read_csv(f"{global_warming_path}/global_warming_dataset.csv")
print("GDP dataset columns:", gdp_df.columns)
print("GDP dataset sample:\n", gdp_df.head())

# Map generic country codes to actual names in GDP data
gdp_df['Country'] = gdp_df['Country'].map(gdp_country_mapping)

# Filter for selected countries GDP data
selected_countries = list(meat_country_mapping.values())
gdp_data = gdp_df[gdp_df['Country'].isin(selected_countries)][['Year', 'GDP', 'Population', 'Country']]

# Add data validation checks
print("GDP data summary:\n", gdp_data.describe())
print("Meat data summary:\n", filtered_meat.describe())

# Verify GDP and Population units
assert (gdp_data['GDP'] > 0).all(), "GDP values must be positive"
assert (gdp_data['Population'] > 0).all(), "Population values must be positive"

# Aggregate GDP data by Year and Country using proper methods
gdp_data = gdp_data.groupby(['Year', 'Country']).agg({
    'GDP': 'sum',  # Use sum instead of mean for GDP
    'Population': 'sum'
}).reset_index()

# Map country codes to names in meat data
filtered_meat['Country'] = filtered_meat['LOCATION'].map(meat_country_mapping)
print("Filtered GDP data:\n", gdp_data.head())

# Calculate GDP per capita in thousands of USD for better readability
gdp_data['GDP_per_capita'] = (gdp_data['GDP'] / gdp_data['Population']) / 1000

# Merge datasets on year and country
data = pd.merge(
    filtered_meat,
    gdp_data,
    left_on=['TIME', 'Country'],
    right_on=['Year', 'Country'],
    how='inner'
)
print("Merged data:\n", data.head())

# Filter relevant columns and rename for clarity
data = data[['Year', 'GDP_per_capita', 'Value', 'Country']]
data = data.rename(columns={'Value': 'Meat_consumption'})

# Filter data starting from 1990
data = data[data['Year'] >= 1990]

# Create output directory if it doesn't exist
import os
os.makedirs('global-warming/output', exist_ok=True)

# Create separate plots for each country
for country in data['Country'].unique():
    country_data = data[data['Country'] == country]
    
    plt.figure(figsize=(10, 8))
    
    # Create color gradient based on year
    colors = plt.cm.viridis((country_data['Year'] - 1990) / (2020 - 1990))
    
    # Create scatter plot with color gradient
    scatter = sns.scatterplot(
        data=country_data,
        x='GDP_per_capita',
        y='Meat_consumption',
        hue='Year',
        palette='viridis',
        size='Year',
        sizes=(50, 200),
        legend='full'
    )
    
    # Add trend line
    sns.regplot(
        data=country_data,
        x='GDP_per_capita',
        y='Meat_consumption',
        scatter=False,
        color='red',
        line_kws={'linestyle':'--', 'linewidth':2}
    )
    
    # Add labels and title
    plt.xlabel('GDP per capita (thousands USD)', fontsize=12)
    plt.ylabel('Meat consumption per capita (kg)', fontsize=12)
    plt.title(f'{country} Meat Consumption vs GDP per Capita (1990-2020)', fontsize=14, pad=20)
    
    # Add colorbar
    norm = plt.Normalize(country_data['Year'].min(), country_data['Year'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Year', ax=scatter.axes)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'global-warming/output/meat_gdp_{country.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
