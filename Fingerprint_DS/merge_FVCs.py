import os
import shutil

# Define the base directory and subdirectories
base_dir = "/home/noshin/Documents/junkbox/Fingerprint/74034_3_En_4_MOESM1_ESM/FVC2004/Dbs"
sub_dirs = ['DB1_A', 'DB1_B', 'DB2_A', 'DB2_B', 'DB3_A', 'DB3_B', 'DB4_A', 'DB4_B']
target_dir = '/home/noshin/Documents/junkbox/Fingerprint/merged'  # Target directory for copied files

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Initialize a counter for renaming .tif files
file_counter = 7041

# Loop through each subdirectory
for subdir in sub_dirs:
    subdir_path = os.path.join(base_dir, subdir)

    # Loop through each file in the subdirectory
    for filename in os.listdir(subdir_path):
        if filename.endswith('.tif'):
            # Construct the full path for the current file
            file_path = os.path.join(subdir_path, filename)

            # Create a new filename for the .tif file (sequential naming)
            new_name = f"image_{file_counter}.tif"
            file_counter += 1

            # Construct the target path for the renamed file
            target_file_path = os.path.join(target_dir, new_name)

            # Copy and rename the file
            shutil.copy2(file_path, target_file_path)
            print(f"Copied and renamed {file_path} to {target_file_path}")

