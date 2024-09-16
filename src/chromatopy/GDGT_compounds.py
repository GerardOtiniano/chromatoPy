def get_gdgt(gdgt_oi):
    # Define the data structures
    ref_struct = {"name": ["Reference"], "GDGT_dict": {"744": "Standard"}, "Trace": ["744"], "window": [10, 30]}
    iso_struct = {"name": ["isoGDGTs"], "GDGT_dict": {"1302": "GDGT-0", "1300": "GDGT-1", "1298": "GDGT-2", "1296": "GDGT-3", "1292": ["GDGT-4", "GDGT-4'"]}, "Trace": ["1302", "1300", "1298", "1296", "1292"], "window": [10, 35]}
    br_struct = {
        "name": ["brGDGTs"],
        "GDGT_dict": {"1050": ["IIIa", "IIIa'"], "1048": ["IIIb", "IIIb'"], "1046": ["IIIc", "IIIc'"], "1036": ["IIa", "IIa'"], "1034": ["IIb", "IIb'"], "1032": ["IIc", "IIc'"], "1022": "Ia", "1020": "Ib", "1018": "Ic"},
        "Trace": ["1050", "1048", "1046", "1036", "1034", "1032", "1022", "1020", "1018"],
        "window": [20, 40],
    }
    oh_struct = {"name": ["OH-GDGTs"], "GDGT_dict": {"1300": "OH-GDGT-0", "1298": ["OH-GDGT-1", "2OH-GDGT-0"], "1296": "OH-GDGT-2"}, "Trace": ["1300", "1298", "1296"], "window": [35, 50]}

    # Map the user's input to the corresponding data structures
    gdgt_map = {"1": iso_struct, "2": br_struct, "3": oh_struct}

    # All GDGT types selected
    if gdgt_oi == "4":
        gdgt_oi = "1,2,3"

    # Initialize the final structure with ref_struct values
    combined_struct = {"name": ref_struct["name"].copy(), "GDGT_dict": ref_struct["GDGT_dict"].copy(), "Trace": ref_struct["Trace"].copy(), "window": ref_struct["window"].copy()}

    # Split the user input and iterate through each selection
    name_list = [ref_struct["name"]]
    gdgt_dict_list = [ref_struct["GDGT_dict"]]
    trace_list = [ref_struct["Trace"]]
    window_list = [ref_struct["window"]]

    if len(gdgt_oi) > 1:
        selected_types = gdgt_oi.split(",")
    else:
        selected_types = gdgt_oi
    for gdgt_type in selected_types:
        gdgt_type = gdgt_type.strip()
        if gdgt_type in gdgt_map:
            struct = gdgt_map[gdgt_type]
            name_list.append(struct["name"])
            gdgt_dict_list.append(struct["GDGT_dict"])
            trace_list.append(struct["Trace"])
            window_list.append(struct["window"])

    return {
        "names": name_list,
        "GDGT_dict": gdgt_dict_list,
        "Trace": trace_list,
        "window": window_list,
    }


def get_gdgt_input():
    # Set of valid combinations
    valid_combinations = {"1", "1,2", "1,2,3", "2", "2,3", "3", "1,3", "4"}

    while True:
        try:
            # Prompt the user for input
            gdgt_oi = input("Enter GDGT types of interest (separate by commas):\n\t1. isoGDGTs\n\t2. brGDGTs\n\t3. OH-GDGTs\n\t4. All\n")

            # Remove spaces and check if the input is in the set of valid combinations
            gdgt_oi = gdgt_oi.replace(" ", "")

            # If the input is not valid, raise a ValueError
            if gdgt_oi not in valid_combinations:
                raise ValueError("Invalid selection. Please enter a valid combination such as '1', '1,2', '1,2,3', etc.")

            # If the input is valid, break the loop and return the input
            return gdgt_oi

        except ValueError as ve:
            # Print the error message and prompt the user again
            print(ve)
