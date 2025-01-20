import json

def load_config(config_path):
    """
    Load a JSON configuration file from the given path.
    :param config_path: The path to the JSON configuration file.
    :return: The configuration as a dictionary.
    """
    with open(config_path, encoding='utf-8') as config_file:
        config = json.load(config_file)
    return config


import os
import json
from dotenv import load_dotenv

load_dotenv()

def load_config(config_path: str) -> dict:
    """
    Load a JSON configuration file from the given path.
    Returns a dictionary with combined environment-based and JSON-based config if needed.
    """
    with open(config_path, encoding="utf-8") as config_file:
        file_config = json.load(config_file)
    
    return file_config
