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

def load_results(output_path, filename="FID_output.json"):
    js_file = os.path.join(output_path, filename)
    if os.path.exists(js_file):
        try:
            with open(js_file, "r") as f:
                return json.load(f)
        except Exception as e:
            tqdm.write("Error loading existing JSON:", e)
    return None

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
    Load an FID integration JSON, delete the given samples, and overwrite the file.

    Parameters
    ----------
    json_path : str
        Path to the existing JSON output from FID_integration().
    to_delete : list of str
        Sample names (keys under data['Samples']) to remove.

    Returns
    -------
    data : dict
        The updated data structure (after deletion).
    """
    # 1) Load
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"No such file: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # 2) Delete requested samples
    for name in to_delete:
        if name in data.get("Samples", {}):
            data["Samples"].pop(name)
        else:
            # silently skip if not present
            continue

    # 3) Write back to the same file
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print("Deleted samples and updated dataset.")