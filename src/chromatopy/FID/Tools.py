import json
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

def save_results(data, output_path):
    js_file = f"{output_path}/FID_output.json"
    os.makedirs(os.path.dirname(js_file), exist_ok=True)
    try:
        with open(js_file, "w") as f:
            json.dump(clean_for_json(data), f, indent=4)
        # tqdm.write(f"Output structure saved to:\n{js_file}")
    except Exception as e:
        tqdm.write("Error saving JSON:", e)

# def load_results(output_path, filename="FID_output.json"):
#     js_file = os.path.join(output_path, filename)
#     if os.path.exists(js_file):
#         try:
#             with open(js_file, "r") as f:
#                 return json.load(f)
#         except Exception as e:
#             tqdm.write("Error loading existing JSON:", e)
#     return None

def load_results(output_path, filename="FID_output.json", list_samples=False, list_processed=False):
    """
    Try to load FID_output.json from output_path.
    If it doesn’t exist, return None.
    Otherwise return the dict, rebuilding any Raw Data dicts into DataFrames.
    """
    js_file = os.path.join(output_path, "FID_output.json")
    if not os.path.exists(js_file):
        return None

    with open(js_file, "r") as f:
        data = json.load(f)

    # rebuild Raw Data dicts into DataFrames
    for sample in data.get("Samples", {}).values():
        raw = sample.get("Raw Data")
        if isinstance(raw, dict):
            sample["Raw Data"] = pd.DataFrame(raw)
    if list_samples:
        for key in data['Samples'].keys():
            print(key)
    if list_processed:
        key = []
        for x in data['Samples'].keys():
            if 'Processed Data' in data['Samples'][x].keys():
                key.append(x)
        print(key)
    return data


def clean_for_json(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.ndarray, pd.Series, list, tuple)):
        return [clean_for_json(el) for el in obj]
    elif isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    else:
        try:
            json.dumps(obj)  # test if serializable
            return obj
        except (TypeError, OverflowError):
            return str(obj)  # fallback

def delete_samples(json_path: str, to_delete: list[str]) -> dict:
    """
    Load an FID integration JSON, delete any samples whose key contains
    one of the given substrings, and overwrite the file.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"No such file: {json_path!r}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # grab the actual dict
    samples: dict = data.get("Samples", {})

    # make a static list of the keys so we can pop safely
    all_keys = list(samples.keys())

    for fragment in to_delete:
        # print(f"Looking for samples containing: {fragment!r}")
        # find matches in that static list
        matches = [k for k in all_keys if fragment in k]
        if not matches:
            tqdm.write(f"Could not delete any samples matching '{fragment}'")
            continue

        for key in matches:
            # actually remove from the dict
            samples.pop(key, None)
            print(f"Deleted sample: {key}  (matched '{fragment}')")

    # write back
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)