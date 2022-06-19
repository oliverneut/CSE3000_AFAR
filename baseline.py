import pandas as pd
from datetime import datetime
from utils import drop_ids
from models import train_xgboost

datasets = {
    'experiment_data/football/': ['football.csv', 'win'],
    'experiment_data/kidney-disease/': ['kidney_disease.csv', 'classification'],
    'experiment_data/steel-plate-fault/': ['steel_plate_fault.csv', 'Class'],
    'experiment_data/titanic/': ['titanic.csv', 'Survived']
}

def baseline():
    results = []
    for path, info in datasets.items():
        file_name = info[0]
        target = info[1]
        df = pd.read_csv(path + file_name)
        df = df.apply(lambda x: pd.factorize(x)[0])          # encode categorical features
        df.drop(drop_ids(df.columns), axis=1, inplace=True)  # drop id columns

        X, y = df.drop(target, axis=1), df[target]

        start_time = datetime.now()
        acc, depth = train_xgboost(X, y)
        end_time = datetime.now()

        time = end_time - start_time
        res = { 
            'table': file_name[:-4],
            'accuracy': acc,
            'depth': depth,
            'runtime': time.total_seconds()
        }
        results.append(res)

    pd.DataFrame(results).to_csv('results/baseline.csv', index=False)


if __name__ == '__main__':
    baseline()