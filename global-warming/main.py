import kagglehub

# Download latest version
path = kagglehub.dataset_download("ankushpanday1/global-warming-dataset-195-countries-1900-2023")

print("Path to dataset files:", path)