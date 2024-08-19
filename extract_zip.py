import zipfile
import os

# Path to the zip file
zip_file_path = r'C:\Users\sowah\OneDrive\Desktop\yrr\ml_belief_2024.zip'

# Directory to extract files to
extract_to_path = 'datasets/movielens'

# Create the directory if it doesn't exist
os.makedirs(extract_to_path, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print("Extraction complete!")
