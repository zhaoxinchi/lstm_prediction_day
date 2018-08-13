

# normalization
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:  # window shape (sequence_length L ,)  即(51L,)
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data