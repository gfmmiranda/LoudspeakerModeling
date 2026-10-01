import numpy as np

def magnitude_to_db(value, reference=1.0):
    value = np.asarray(value)
    return 20 * np.log10(np.abs(value) / reference)