#read 
from hashlib import new
import json
import random
import rootutils
import os

rootutils.setup_root(__file__, indicator=".project-root")
ROOT = os.environ.get("PROJECT_ROOT")
# ROOT = os.path.join(ROOT, "LLaVA", "textcaps")
# ROOT = "/home/victorletzelter/workspace/LoRA-MCL_cleaned/"

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: File at {file_path} is not valid JSON")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

# Path to the JSON file

json_files_dict = {
f"{os.path.join(ROOT, 'LLaVA', 'textcaps', 'TextCaps_0.1_train.json')}": f"{os.path.join(ROOT, 'LLaVA', 'playground', 'data', 'textCapsTrain.json')}",
f"{os.path.join(ROOT, 'LLaVA', 'textcaps', 'TextCaps_0.1_val.json')}": f"{os.path.join(ROOT, 'LLaVA', 'playground', 'data', 'textCapsVal.json')}",
}

for json_file in json_files_dict.keys():

    json_file_path = json_file
    new_path = json_files_dict[json_file]

    # Read the JSON data
    data = read_json_file(json_file_path)

    if data is not None:
        # Print some basic info about the loaded data
        if isinstance(data, dict):
            print(f"Successfully loaded JSON data with {len(data)} top-level keys")
            print("Top-level keys:", list(data.keys()))
        elif isinstance(data, list):
            print(f"Successfully loaded JSON data with {len(data)} items")
            if len(data) > 0:
                print("First item sample:", data[0])
        else:
            print("Successfully loaded JSON data")

    # Create a dictionary to store reference strings for each image_id
    id_to_refs = {}

    # Iterate through the data to collect reference strings by image_id
    for item in data['data']:
        image_id = item['image_id']
        refs = tuple(item['reference_strs'])  # Convert list to tuple for hashability
        
        if image_id not in id_to_refs:
            id_to_refs[image_id] = {'refs': refs, 'item': item}
        else:
            # Check if the reference strings match
            if refs != id_to_refs[image_id]['refs']:
                print(f"Warning: Found different reference strings for image_id {image_id}")
                print("First set:", id_to_refs[image_id]['refs'])
                print("Second set:", refs)

    # Create new data structure without duplicates
    data2 = {'data': []}
    for image_id, info in id_to_refs.items():
        data2['data'].append(info['item'])

    print(f"Original data length: {len(data['data'])}")
    print(f"Deduplicated data length: {len(data2['data'])}")

    # Create a dictionary to store reference strings for each image_id
    id_to_refs = {}

    # Iterate through the data to collect reference strings by image_id
    for item in data['data']:
        image_id = item['image_id']
        refs = tuple(item['reference_strs'])  # Convert list to tuple for hashability
        
        if image_id not in id_to_refs:
            id_to_refs[image_id] = {'refs': refs, 'item': item}
        else:
            # Check if the reference strings match
            if refs != id_to_refs[image_id]['refs']:
                print(f"Warning: Found different reference strings for image_id {image_id}")
                print("First set:", id_to_refs[image_id]['refs'])
                print("Second set:", refs)

    # Create new data structure without duplicates
    data2 = {'data': []}
    for image_id, info in id_to_refs.items():
        data2['data'].append(info['item'])

    print(f"Original data length: {len(data['data'])}")
    print(f"Deduplicated data length: {len(data2['data'])}")

    # Create new formatted data structure with all references
    formatted_data_all_refs = {'data': []}

    for item in data2['data']:
        # Extract image ID and all reference strings
        image_id = item['image_id']
        references = item['reference_strs']
        
        # Create new formatted item for each reference
        for reference in references:
            new_item = {
                "id": str(image_id),
                "image": f"{image_id}.jpg",
                "conversations": [
                    {
                        "from": "human", 
                        "value": "<image>\nPlease carefully observe the image and come up with a caption for the image."
                    },
                    {
                        "from": "gpt",
                        "value": reference
                    }
                ]
            }
            formatted_data_all_refs['data'].append(new_item)

    # Shuffle the data
    random.shuffle(formatted_data_all_refs['data'])

    data3 = formatted_data_all_refs

    print(f"Reformatted {len(data3['data'])} items")

    # Create new formatted data structure with all references
    formatted_data_all_refs = {'data': []}

    for item in data2['data']:
        # Extract image ID and all reference strings
        image_id = item['image_id']
        references = item['reference_strs']
        
        # Create new formatted item for each reference
        for reference in references:
            new_item = {
                "id": str(image_id),
                "image": f"{image_id}.jpg",
                "conversations": [
                    {
                        "from": "human", 
                        "value": "<image>\nPlease carefully observe the image and come up with a caption for the image."
                    },
                    {
                        "from": "gpt",
                        "value": reference
                    }
                ]
            }
            formatted_data_all_refs['data'].append(new_item)

    # Shuffle the data
    random.shuffle(formatted_data_all_refs['data'])

    data3 = formatted_data_all_refs

    print(f"Reformatted {len(data3['data'])} items")

    with open(new_path, 'w') as f:
        json.dump(data3['data'], f)

