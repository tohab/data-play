import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download datasets
global_warming_path = kagglehub.dataset_download("ankushpanday1/global-warming-dataset-195-countries-1900-2023")
meat_consumption_path = kagglehub.dataset_download("vagifa/meatconsumption")

print("Global warming dataset path:", global_warming_path)
print("Meat consumption dataset path:", meat_consumption_path)

# Load global warming data
df = pd.read_csv(f"{global_warming_path}/global_warming_dataset.csv")

# Get first 4 unique countries for analysis
focus_countries = df['Country'].unique()[:4]
df = df[df['Country'].isin(focus_countries)]
print(f"\nAnalyzing countries: {focus_countries}")

# Basic analysis
print("\n=== Dataset Info ===")
print(df.info())

print("\n=== Summary Statistics ===")
print(df.describe())

# Visualizations
plt.figure(figsize=(12, 6))

# Temperature trends over time
plt.subplot(1, 2, 1)
for country in focus_countries:
    country_data = df[df['Country'] == country]
    country_data.groupby('Year')['Average_Temperature'].mean().plot(label=country)
plt.title('Temperature Trends: Sample Countries')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')

# Temperature distribution by country
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='Average_Temperature', hue='Country', kde=True, element='step')
plt.title('Temperature Distribution: Sample Countries')
plt.xlabel('Temperature (°C)')

# Create output directory
import os
os.makedirs('global-warming/output', exist_ok=True)

plt.tight_layout()
plt.savefig('global-warming/output/temperature_analysis.png')

# Correlation analysis
print("\n=== Correlation Matrix ===")
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)

# Heatmap visualization
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap: Sample Countries')
plt.savefig('global-warming/output/correlation_heatmap.png')

# Kuznets curve analysis
df['GDP_per_capita'] = df['GDP'] / df['Population']

# Create figure for Kuznets analysis
plt.figure(figsize=(12, 6))

# Scatter plot with quadratic fit
plt.subplot(1, 2, 1)
sns.scatterplot(x='GDP_per_capita', y='Fossil_Fuel_Usage', hue='Country', data=df)
sns.regplot(x='GDP_per_capita', y='Fossil_Fuel_Usage', data=df, 
            order=2, scatter=False, line_kws={'color':'black'})
plt.xscale('log')
plt.title('Fossil Fuel Usage vs GDP: Sample Countries')
plt.xlabel('GDP per Capita (log scale)')
plt.ylabel('Fossil Fuel Usage')

# Temporal analysis by country
df['Decade'] = (df['Year'] // 10) * 10
plt.subplot(1, 2, 2)
sns.lineplot(x='GDP_per_capita', y='Fossil_Fuel_Usage', hue='Country',
             data=df, estimator='mean', ci=None)
plt.xscale('log')
plt.title('Fossil Fuel Usage: Sample Countries')
plt.xlabel('GDP per Capita (log scale)')
plt.ylabel('Average Fossil Fuel Usage')

plt.tight_layout()
plt.savefig('global-warming/output/kuznets_analysis.png')
