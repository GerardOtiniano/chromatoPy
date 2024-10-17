# src/chromatopy/utils/errors/smoothing_check.py

def smoothing_check(smoothing_window, smoothing_factor):
    """
    Validates that the smoothing_window is at least one higher than the smoothing_factor.
    If the condition is not met, prompts the user to re-enter both values until valid inputs are provided.

    Parameters
    ----------
    smoothing_window : int
        The window size for smoothing.
    smoothing_factor : int
        The factor used for smoothing.

    Returns
    -------
    tuple
        A tuple containing the validated (smoothing_window, smoothing_factor).

    Raises
    ------
    ValueError
        If the user fails to provide valid integer inputs after multiple attempts.
    """
    while smoothing_window <= smoothing_factor:
        print("\nError: The smoothing window must be at least 1 higher than the smoothing factor.")
        
        # Prompt the user for new smoothing_window
        try:
            smoothing_window_input = input("Enter revised value for smoothing_window: ")
            smoothing_window = int(smoothing_window_input)
        except ValueError:
            print("Invalid input for smoothing_window. Please enter a valid integer.")
            continue  # Restart the loop to prompt again
        
        # Prompt the user for new smoothing_factor
        try:
            smoothing_factor_input = input("Enter revised value for smoothing_factor: ")
            smoothing_factor = int(smoothing_factor_input)
        except ValueError:
            print("Invalid input for smoothing_factor. Please enter a valid integer.")
            continue  # Restart the loop to prompt again
        
        # Additional check to ensure smoothing_window is now greater than smoothing_factor
        if smoothing_window <= smoothing_factor:
            print("The smoothing window must still be at least 1 higher than the smoothing factor. Please try again.")
    
    print(f"\nValidated Inputs:\nSmoothing Window: {smoothing_window}\nSmoothing Factor: {smoothing_factor}")
    return {"sw": smoothing_window, 
            "sf": smoothing_factor}
