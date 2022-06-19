import pandas as pd
from datetime import datetime
from utils import drop_ids
from models import train_xgboost
from feature_selection import feature_selection

datasets = {
    'experiment_data/football/': 'win',
    'experiment_data/kidney-disease/': 'classification',
    'experiment_data/steel-plate-fault/': 'Class',
    'experiment_data/titanic/': 'Survived'
}

methods = ['info_gain', 'pearson', 'spearman', 'gini', 'symetrical_uncertainty']

def dummy():
    results = []
    for path, target in datasets.items():
        print(path)
        start_time = datetime.now()
        connections = pd.read_csv(path + 'connections.csv')
        dummy_table = pd.read_csv(path + connections['from_table'][0])

        for index, row in connections.iterrows():
            pk, fk = row['from_key'], row['to_key']
            to_table = pd.read_csv(path + row['to_table'])
            dummy_table = pd.merge(dummy_table, to_table, how='left', left_on=pk, right_on=fk)
        
        dummy_table.drop(drop_ids(dummy_table.columns), axis=1, inplace=True)
        dummy_table = dummy_table.apply(lambda x: pd.factorize(x)[0])
        
        print(dummy_table.shape)

        X, y = dummy_table.drop(target, axis=1), dummy_table[target]
        
        acc, depth = train_xgboost(X, y)
        end_time = datetime.now()
        time = end_time - start_time

        res = {
            'dataset': connections['from_table'][0][:-4],
            'accuracy': acc,
            'depth': depth,
            'runtime': time.total_seconds()
        }
        results.append(res)

    pd.DataFrame(results).to_csv('results/dummy.csv', index=False)

def dummy_feature_selection():
    results = []

    for method in methods:
        for path, target in datasets.items():
            print(path)
            start_time = datetime.now()
            connections = pd.read_csv(path + 'connections.csv')
            dummy_table = pd.read_csv(path + connections['from_table'][0])

            for index, row in connections.iterrows():
                pk = row['from_key']
                fk = row['to_key']
                to_table = pd.read_csv(path + row['to_table'])
                dummy_table = pd.merge(dummy_table, to_table, how='left', left_on=pk, right_on=fk)
            
            dummy_table.drop(drop_ids(dummy_table.columns), axis=1, inplace=True)
            dummy_table = dummy_table.apply(lambda x: pd.factorize(x)[0])

            X, y = dummy_table.drop(target, axis=1), dummy_table[target]

            # feature selection
            total_cols = X.shape[1]
            k = round(total_cols / 2)
            cols = feature_selection(method, X, y, k)
            acc, depth = train_xgboost(X[cols], y)
            end_time = datetime.now()
            time = end_time - start_time

            res = {
                'dataset': connections['from_table'][0][:-4],
                'filter method': method,
                'accuracy': acc,
                'depth': depth,
                'runtime': time.total_seconds()
            }
            results.append(res)

    pd.DataFrame(results).to_csv('results/dummy_feature_selection.csv', index=False)

if __name__ == '__main__':
    dummy_feature_selection()
