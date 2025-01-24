import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import os

# Download meat consumption dataset
print("Downloading dataset...")
meat_consumption_path = kagglehub.dataset_download("scibearia/meat-consumption-per-capita")
# Load meat consumption data with error handling
try:
    meat_df = pd.read_csv(f"{meat_consumption_path}/Consumption of meat per capita.csv")
except FileNotFoundError:
    print("Error: Could not find the expected CSV file in the downloaded dataset")
    print("Available files:")
    import os
    print(os.listdir(meat_consumption_path))
    exit(1)

# Filter for United States data
print("Processing data...")
us_data = meat_df[meat_df['Entity'] == 'United States']

# Extract relevant meat types and years
meat_types = ['Poultry', 'Beef', 'Sheep and goat', 'Pork']
years = range(1961, 2022)
us_data = us_data[(us_data['Year'] >= 1961) & (us_data['Year'] <= 2021)]

# Create output directory if it doesn't exist
os.makedirs('global-warming/output', exist_ok=True)

# Create stacked area plot
print("Creating visualization...")
plt.figure(figsize=(12, 8))

# Prepare data for stacking
stack_data = [us_data[meat] for meat in meat_types]

# Create plot
plt.stackplot(years, stack_data, labels=meat_types, alpha=0.8)

# Add plot elements
plt.title('US Meat Consumption Breakdown (1961-2021)', fontsize=16, pad=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Meat Consumption per Capita (kg)', fontsize=14)
plt.legend(title='Meat Type', fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(range(1960, 2025, 5))

# Save plot
print("Saving visualization...")
plt.tight_layout()
plt.savefig('global-warming/output/us_meat_consumption_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved to global-warming/output/us_meat_consumption_breakdown.png")
