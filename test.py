import os
import pickle

filename = 'random_forest_model.pkl'
current_directory = os.path.dirname(__file__)
full_path = os.path.join(current_directory, 'models', filename)

try:
    with open(full_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except pickle.UnpicklingError:
    print("Error: Failed to unpickle the file. It might not be a valid pickle file.")
except Exception as e:
    print("An error occurred:", e)
