import os

# Replace 'path/to/folder' with the path to your folder
folder_path = "archive//images"

# Open a text file for writing
with open('file_names.txt', 'w') as file:
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Write the file name to the text file
        file.write(filename + '\n')