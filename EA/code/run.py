import numpy as np 
import random
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from time import time
from packing import Individual, Population, Circle, Rect


import argparse
import json

import pickle

random.seed(0)
np.random.seed(0)


def load_data(filename):
    path = Path("../data") / str("r" + filename + ".csv")
    data = pd.read_csv(path, header=None)
    data.columns = ["width", "height", "value"]
    return data

def parse_config(config_filepath):
    config_file = json.load(open(config_filepath))
    try:
        experiment_name = config_file["experiment_name"]
        dataset_name = config_file["dataset_name"]
        experiment_filepath = config_file["experiment_filepath"]
        reruns = config_file["reruns"]
        hyperparameters = config_file["hyperparameters"]
    except KeyError:
        raise KeyError("Config file must contain keys experiment_name, dataset_name, experiment_filepath and reruns")
    
    return config_file



def run(config_file):
    
    data = load_data(config_file["dataset_name"])
    
    Path(config_file["experiment_filepath"] / config_file["experiment_name"]).mkdir(parents=True, exist_ok=True)
    
    r = int(config_file["dataset_name"])
    
    best_individual_scores_accross_runs = {}
    avg_individual_scores_accross_runs = {}
    times_accross_runs = {}
    
    for run in range(config_file["reruns"]):
        temperature = config_file.get("temperature", 1000)
        population = Population(r, data, temperature=temperature)

        best_individual_scores, avg_individual_scores, times = population.run(**config_file["hyperparameters"])

        best_individual_scores_accross_runs[run] = best_individual_scores
        avg_individual_scores_accross_runs[run] = avg_individual_scores
        times_accross_runs[run] = times

        with open(config_file["experiment_filepath"] / config_file["experiment_name"] / str(run + ".pkl"), "wb") as f:
            pickle.dump((best_individual_scores, avg_individual_scores, times, population), f)

    best_individual_scores_accross_runs = pd.DataFrame(best_individual_scores_accross_runs)
    avg_individual_scores_accross_runs = pd.DataFrame(avg_individual_scores_accross_runs)
    times_accross_runs = pd.DataFrame(times_accross_runs)
    
    best_individual_scores_accross_runs.to_csv(config_file["experiment_filepath"] / config_file["experiment_name"] / "best_individual_scores.csv")
    avg_individual_scores_accross_runs.to_csv(config_file["experiment_filepath"] / config_file["experiment_name"] / "avg_individual_scores.csv")
    times_accross_runs.to_csv(config_file["experiment_filepath"] / config_file["experiment_name"] / "times.csv")
    
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("--config", type=str, required=True)
    
    args = argparser.parse_args()
    config_file = parse_config(args.config)
    run(config_file)
    print("koniec")
    