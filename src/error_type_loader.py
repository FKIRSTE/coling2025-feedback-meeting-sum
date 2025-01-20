import json
import os
from typing import Dict

def build_from_files(directory_path: str = "src/error_types") -> Dict[str, dict]:
    """
    Load all .json error definition files from the given directory and
    build a dictionary keyed by the file name (minus .json).
    """
    rebuilt_dict = {}
    if not os.path.isdir(directory_path):
        return rebuilt_dict

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".json"):
            key = file_name[:-5]  # remove '.json'
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            rebuilt_dict[key] = data
    return rebuilt_dict
