import json

# Replace 'path_to_file.json' with the actual file path
file_path = '/home/hongji/Downloads/MPI_annot.json'

# Open the JSON file for reading
with open(file_path, 'r') as file:
    # Parse JSON data from the file
    data = json.load(file)

    print('reading file')

# Now 'data' holds the parsed JSON data
print(data)


