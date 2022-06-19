# CSE3000_AFAR
Code of experiments and evaluation


## Data
The main.py file contains a function preprocess, that modifies parts of the data for performing the evaluation. Make sure to run this, before running the experiments.

Preprocessing of the data:
  1. Fix target of steel_plate_fault base table (1's and 2's -> 0's and 1's)
  2. Kidney disease has 2 misclassified values in target ('ckd\t' -> 'ckd')
  3. Rename the columns of the candidate tables and the keys in connections.csv files.

Experiments:
- baseline.py
- dummy.py
- approach.py



Datasets from kaggle:
- https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists
- https://www.kaggle.com/datasets/mastmustu/income
- https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
- https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
- https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
- https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
- https://www.kaggle.com/datasets/mojtaba142/hotel-booking
- https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
- https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
