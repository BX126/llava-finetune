from datasets import load_dataset
from PIL import Image
from io import BytesIO
import requests
import os
import json
import uuid

def process_and_save(dataset, output_folder, subset_name):
    subset_folder = os.path.join(output_folder, subset_name)
    image_subfolder = os.path.join(output_folder, 'images')

    if not os.path.exists(image_subfolder):
        os.makedirs(image_subfolder)

    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)

    json_data_list = []

    for item in dataset:
        if isinstance(item['image'], str):
            response = requests.get(item['image'])
            image = Image.open(BytesIO(response.content))
        else:
            image = item['image'] 
        unique_id = str(uuid.uuid4())
        image_path = os.path.join(image_subfolder, f"{unique_id}.jpg")


        # Save image
        image.save(image_path)


        # Remove duplicates and format answers
        answers = item['answers']
        unique_answers = list(set(answers))
        formatted_answers = ", ".join(unique_answers)


        # Structure for LLaVA JSON
        json_data = {
            "id": unique_id,
            "image": f"{unique_id}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": item['question']
                },
                {
                    "from": "gpt",
                    "value": formatted_answers
                }
            ]
        }


        # Append to list
        json_data_list.append(json_data)


    # Save the JSON data list to a file
    json_output_path = os.path.join(output_folder, subset_name, 'dataset.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)


if __name__ == "__main__":
    dataset = load_dataset("vqa_v2", "val")
    process_and_save(dataset, "data", "val")