import numpy as np 
import random
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from time import time
from networks import Population

from sklearn import datasets
from sklearn.model_selection import train_test_split


import argparse
import json

import pickle

random.seed(0)
np.random.seed(0)
import os


def load_data(filename, random_state=0):
    if filename == "iris":
        iris = datasets.load_iris()
        X = iris["data"]
        y = iris["target"].reshape(-1, 1)
        
        # divide to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        
        return X_train, y_train, X_test, y_test
    
    if filename == "multimodal":
        
        data_dir = Path("../data")
        
        multimodal_train = pd.read_csv(data_dir / "multimodal-large-training.csv", index_col=False)
        multimodal_test = pd.read_csv(data_dir / "multimodal-large-test.csv", index_col=False)
        X_train = multimodal_train[["x"]].to_numpy()
        X_test = multimodal_test[["x"]].to_numpy()
        y_train =  multimodal_train[["y"]].to_numpy()
        y_test = multimodal_test[["y"]].to_numpy()
        
        X = np.vstack([X_train, X_test])
        y = np.vstack([y_train, y_test])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        return X_train, y_train, X_test, y_test
    
    if filename == "auto-mpg":
        data = pd.read_csv("../data/auto-mpg.data", delim_whitespace=True, header=None)
        data.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
        X = data.iloc[:, 1:-1].to_numpy()
        y = data.iloc[:, 0].to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        return X_train, y_train, X_test, y_test
        
    
    
def load_layers(problem_type):
    if problem_type == "classification":
        layers = [
            {"output_dim": 5, "activation": "relu", "init": "he"},
            {"output_dim": 5, "activation": "relu", "init": "he"},
            {"output_dim": 3, "activation": "softmax", "init": "normal"}
        ]
    elif problem_type == "regression":
        layers = [
            {"output_dim": 20, "activation": "relu", "init": "he"},
            {"output_dim": 20, "activation": "relu", "init": "he"},
            {"activation": "linear", "init": "normal"}
        ]
    else:
        raise ValueError("Problem type must be either classification or regression")
    
    return layers


def parse_config(config_filepath):
    config_file = json.load(open(config_filepath))
    try:
        experiment_name = config_file["experiment_name"]
        dataset_name = config_file["dataset_name"]
        config_file["experiment_filepath"] = Path(config_file["experiment_filepath"])
        reruns = config_file["reruns"]
        hyperparameters = config_file["hyperparameters"]
    except KeyError:
        raise KeyError("Config file must contain keys experiment_name, dataset_name, experiment_filepath and reruns")
    
    return config_file



def run(config_file, force = False):
    
    layers = load_layers(config_file["problem_type"])
    
    Path(Path(config_file["experiment_filepath"]) / config_file["experiment_name"]).mkdir(parents=True, exist_ok=True)
    # check if there are files already in the directory
    if len(os.listdir(Path(config_file["experiment_filepath"]) / config_file["experiment_name"])) > 0 and not force:
        raise ValueError("Directory is not empty")
    
    
    
    best_individual_scores_accross_runs = {}
    avg_individual_scores_accross_runs = {}
    
    for run in range(config_file["reruns"]):
        X_train, y_train, X_test, y_test = load_data(config_file["dataset_name"], random_state=run)
        population = Population(layers=layers, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, problem_type=config_file["problem_type"])
        
        population.generate_individuals(config_file["n_indviduals"])

        best_individual_scores, avg_individual_scores = population.run(**config_file["hyperparameters"])

        best_individual_scores_accross_runs[run] = best_individual_scores
        avg_individual_scores_accross_runs[run] = avg_individual_scores

        with open(Path(config_file["experiment_filepath"]) / config_file["experiment_name"] / str(str(run) + ".pkl"), "wb") as f:
            pickle.dump((best_individual_scores, avg_individual_scores, population), f)
            
        print(f"Run {run} finished")
        print(f"Best individual score: {best_individual_scores[-1]}")
        print(f"Avg individual score: {avg_individual_scores[-1]}")

    best_individual_scores_accross_runs = pd.DataFrame(best_individual_scores_accross_runs)
    avg_individual_scores_accross_runs = pd.DataFrame(avg_individual_scores_accross_runs)
    
    best_individual_scores_accross_runs.to_csv(config_file["experiment_filepath"] / config_file["experiment_name"] / "best_individual_scores.csv")
    avg_individual_scores_accross_runs.to_csv(config_file["experiment_filepath"] / config_file["experiment_name"] / "avg_individual_scores.csv")
    
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument("--force", action="store_true")
    
    args = argparser.parse_args()
    config_file = parse_config(args.config)
    run(config_file, force = args.force)
    print("koniec")
    
