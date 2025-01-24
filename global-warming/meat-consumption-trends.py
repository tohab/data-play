import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download meat consumption dataset
meat_consumption_path = kagglehub.dataset_download("vagifa/meatconsumption")
meat_df = pd.read_csv(f"{meat_consumption_path}/meat_consumption_worldwide.csv")

# Country mapping
meat_country_mapping = {
    'USA': 'United States',
    'CHN': 'China',
    'IND': 'India',
    'GBR': 'United Kingdom'
}

# Filter for selected countries
countries = list(meat_country_mapping.keys())
filtered_meat = meat_df[meat_df['LOCATION'].isin(countries)]

# Map country codes to names
filtered_meat['Country'] = filtered_meat['LOCATION'].map(meat_country_mapping)

# Create output directory if it doesn't exist
os.makedirs('global-warming/output', exist_ok=True)

# Create plot
plt.figure(figsize=(12, 8))

# Plot meat consumption over time for each country
for country in filtered_meat['Country'].unique():
    country_data = filtered_meat[filtered_meat['Country'] == country]
    plt.plot(country_data['TIME'], country_data['Value'], label=country, linewidth=2.5)

# Add plot elements
plt.title('Meat Consumption Over Time (1990-2020)', fontsize=16, pad=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Meat Consumption per Capita (kg)', fontsize=14)
plt.legend(title='Country', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(range(1990, 2021, 5))

# Save plot
plt.tight_layout()
plt.savefig('global-warming/output/meat_consumption_trends.png', dpi=300, bbox_inches='tight')
plt.close()
