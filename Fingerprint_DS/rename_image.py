import os

# Define the folder containing the images
folder_path = '/home/noshin/Downloads/collected_data-20250123T030607Z-001/collected_data/pair_data'

# Specify the desired file extension for renamed images
file_extension = '.png'  # Change this if needed (e.g., '.jpg')

# Get a list of all files in the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]

# Sort the files to ensure they are renamed in order
file_list.sort()

# Rename files sequentially
for i, filename in enumerate(file_list, start=1):
    # Construct the full path to the current file
    old_file_path = os.path.join(folder_path, filename)
    
    # Create the new filename
    new_filename = f"{i:04d}{file_extension}"  # e.g., 0001.png, 0002.png
    new_file_path = os.path.join(folder_path, new_filename)
    
    # Rename the file
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {filename} -> {new_filename}")

print("All files have been renamed sequentially.")
