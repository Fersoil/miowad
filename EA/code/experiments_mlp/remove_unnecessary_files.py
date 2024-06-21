import numpy as np

import sys
sys.path.append("..")
import networks

from networks import Population
import pickle


import os
import glob

def list_pkl_files(directory):
    # Use glob to recursively find all .pkl files
    pkl_files = glob.glob(os.path.join(directory, '**', '*.pkl'), recursive=True)
    return pkl_files

# Specify the directory to search
directory_to_search = '.'

# List all .pkl files recursively
pkl_files = list_pkl_files(directory_to_search)

# Print the list of .pkl files
for file in pkl_files:
    try:
        print(f"Extracting file {file}")
        filename = file

        with open(filename, "rb") as f:
            pop = pickle.load(f)[2]


        best_individual = pop.individuals[np.argmin(pop.get_scores())]

        with open(filename, "wb") as f:
            pickle.dump(best_individual, f)
    except:
        print(f"Failed opening file {file}")
