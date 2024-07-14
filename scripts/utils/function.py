import subprocess
import csv
import os
import re
import json
import argparse

def save_args(args, path):
    args_dict = vars(args)
    with open(path, 'w') as f:
        json.dump(args_dict, f)
        
def load_args(path):
    with open(path, 'r') as f:
        args_dict = json.load(f)
    
    args = argparse.Namespace(**args_dict)
    return args

def append_image_to_csv(image, csv_file_path, separator='\t'):
    # Check if the file exists and if we need to write headers
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=separator)
        
        # If the file didn't exist, write the header row first
        if not file_exists:
            writer.writerow(["Image", "Target_Question", "Reference_Answer", "Injected_Prompt", "Attacked_Answer"])

        writer.writerow([
            image.id,
            repr(image.target_question),
            repr(image.reference_answer),
            repr(image.injected_prompt),
            repr('|-----|'.join(image.target_answer) if isinstance(image.target_answer, list) else image.target_answer),
        ])

def filter_list(directory):
    numbers = []
    for filename in os.listdir(directory):
        match = re.search(r'\d+', filename)
        if match:
            number = int(match.group())
            numbers.append(number)
    return numbers