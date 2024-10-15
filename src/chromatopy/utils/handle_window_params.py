# src/chromatopy/utils/handle_window_params.py
from .messages import window_instructions

def hand_window_params(windows, default_windows, gdgt_meta_set):
    """
    Handles the selection and validation of time windows for GDGT (Glycerol Dialkyl Glycerol Tetraethers) groups.
    
    This function allows for either using default windows or prompting the user to input custom time windows for each GDGT group. The windows represent the retention times for specific GDGT compounds in chromatographic data.
    
    Parameters
    ----------
    windows : bool or list
        If True, the default windows will be used.
        If False, the user will be prompted to enter custom time windows for each GDGT group.
        If a list is provided, it is assumed to be custom windows. The function will validate that the list matches the number of GDGT groups.
    default_windows : list
        A list of default time windows for each GDGT group, used if windows is True or no valid input is provided.
    gdgt_meta_set : dict
        A dictionary containing metadata for each GDGT group, including names, traces, and GDGT compound mappings.
    
    Returns
    -------
    dict
        A dictionary containing:
        - "windows" (list): The selected or custom windows for each GDGT group.
        - "GDGT_dict" (dict): The mapping of trace IDs to GDGT compound names for each GDGT group.
        - "trace_ids" (list): A flattened list of trace IDs for all GDGT groups.
    
    Raises
    ------
    ValueError
        If the number of custom windows provided does not match the number of GDGT groups selected.
    
    Example
    -------
    If windows is False, the user is prompted to input time windows for each GDGT group:
    
    Enter new window for isoGDGTs: 10.5, 12.0
    
    Returns:
    {
        "windows": [[10.5, 12.0], [20, 40], [35, 50]],
        "GDGT_dict": {...},
        "trace_ids": [...]
    }
    """
    # Handle custom windows
    if windows:
        # Use default windows
        windows = default_windows
    elif windows is False:
        # Prompt user to input custom windows
        windows = []
        window_instructions()
        for idx, (gdgt_group, default_window) in enumerate(zip(gdgt_meta_set["names"], default_windows)):
            print(f"{idx + 1}. {gdgt_group}: {default_window}")
            # Prompt user for new window
            user_input = input(f"Enter new window for {gdgt_group} as two numbers separated by a comma (e.g., 10.5,12.0): ")
            try:
                lower, upper = map(float, user_input.split(","))
                windows.append([lower, upper])
            except ValueError:
                print("Invalid input. Please enter two numbers separated by a comma.")
                # You might want to handle retries or set default
                windows.append(default_window)  # Use default if invalid
    else:
        # Validate the provided windows
        if len(windows) != len(gdgt_meta_set["names"]):
            raise ValueError("The number of custom windows provided does not match the number of GDGT groups selected.")
    # windows = gdgt_meta_set["window"]
    GDGT_dict = gdgt_meta_set["GDGT_dict"]
    trace_ids = [x for trace in gdgt_meta_set["Trace"] for x in trace]
    return {
        "windows": windows,
        "GDGT_dict": GDGT_dict,
        "trace_ids": trace_ids}
