import os
import re
import pandas as pd
from datasets import load_dataset, Features, Value
from typing import Optional, List, Dict


def find_csv_filenames(path_to_dir: str, suffix: str = ".csv") -> List[str]:
    filenames = os.listdir(path_to_dir)
    return [
        os.path.join(path_to_dir, filename)
        for filename in filenames
        if filename.endswith(suffix)
    ]



def get_dataset(file: str, features: Features, format: str = "csv", delimiter: str = ","):
    """
    Returns a HuggingFace Dataset from a single CSV file, filtering out missing lines.
    """
    dataset = load_dataset(
        format,
        data_files=[file],
        delimiter=delimiter,
        features=features
    )

    def filter_missing_values(example):
        return all(value is not None and value != "" for value in example.values())

    filtered_dataset = dataset["train"].filter(filter_missing_values)
    dataset["train"] = filtered_dataset
    return dataset


def get_dataframe(file_path: str, delimiter: str = ",", limit: Optional[int] = 10) -> pd.DataFrame:
    """
    Concatenate all CSV files in a directory, or read directly if file_path is a CSV.
    Then optionally reduce to a limited number of rows (for debugging).
    """
    if os.path.isdir(file_path):
        csv_files = find_csv_filenames(file_path, suffix=".csv")
        dfs = [pd.read_csv(csv_file, sep=delimiter) for csv_file in csv_files]
        combined = pd.concat(dfs, ignore_index=True)
    else:
        combined = pd.read_csv(file_path, sep=delimiter)

    if limit is not None and limit > 0:
        combined = combined.head(limit)
    return combined


def remove_section(text: str, section_name: str) -> str:
    pattern = re.compile(rf"## {section_name}:(.*?)(!!\n|$)", re.DOTALL)
    return re.sub(pattern, "", text)


def split_feedback_into_dict(feedback: str, remove_content: Optional[list] = None) -> Dict[str, str]:
    """
    Splits the feedback into blocks that start with `## Error type:`.
    Optionally remove specified sections from each block.
    """
    if remove_content is None:
        remove_content = []

    blocks = re.split(r"(?=## Error type:)", feedback)
    feedback_dict = {}

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Locate the error-type line
        start_index = block.find("## Error type:")
        if start_index == -1:
            continue
        end_index = block.find("!!", start_index)
        if end_index == -1:
            continue

        # Extract the error type
        error_type_line = block[start_index:end_index]
        error_type = error_type_line.replace("## Error type:", "").strip()

        # The rest is the block content
        content = block[end_index:].strip()

        # Remove unwanted sections
        for section_name in remove_content:
            content = remove_section(content, section_name)

        # Ensure it ends with '!!'
        if not content.endswith("!!"):
            content += " !!"

        feedback_dict[error_type] = content
    return feedback_dict

def assemble_feedback_from_dict(feedback_dict: Dict[str, str]) -> str:
    """
    Rebuild a single string from the feedback_dict where keys are error types.
    """
    feedback_text = ""
    for error_type, content in feedback_dict.items():
        feedback_text += f"## Error type: {error_type} !!\n"
        if not content.strip().endswith("!!"):
            content = content.strip() + " !!"
        feedback_text += content + "\n\n"
    return feedback_text
