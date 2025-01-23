import os

# Define the parent directory containing the folders
# parent_directory = "C:\\Users\\Noshin\\OneDrive\\Desktop\\faria\\fingerprint\\Collected_Data"
parent_directory = "/home/noshin/JunkBox/fingerprint/collected_data"


# List of folder names
folders = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

for folder in folders:
    folder_path = os.path.join(parent_directory, folder)
    if os.path.isdir(folder_path):
        # List all files in the folder
        files = os.listdir(folder_path)
        # Filter to include only image files (e.g., .png, .jpg, .jpeg)
        image_files = [f for f in files if f.lower().endswith(('.png'))]

        for index, filename in enumerate(image_files, start=1):
            # Generate new file name as "1.png", "2.png", etc.
            new_name = f"{index:04d}.png"
            # Get full paths
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_name)
            # Rename file
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} -> {new_file}")

    print("Renaming complete.")
