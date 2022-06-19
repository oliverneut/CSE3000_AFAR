import pandas as pd
from models import train_xgboost

datasets = {
    "experiment_data/football/": "win",
    "experiment_data/kidney-disease/": "classification",
    "experiment_data/steel-plate-fault/": "Class",
    "experiment_data/titanic/": "Survived"
}

def rename(path, connections):
    base = connections['from_table'][0]
    for index, row in connections.iterrows():
        to_table = pd.read_csv(path + row['to_table'])
        cols = to_table.columns
        to_table.columns = [col + '_' + row['to_table'][:-4] for col in cols]
        to_table.to_csv(path + row['to_table'], index=False)
        
        pk, fk = row['from_key'], row['to_key']

        if base != row['from_table']:
            connections['from_key'][index] = pk +'_'+ row['from_table'][:-4]
        connections['to_key'][index] = fk +'_'+ row['to_table'][:-4]
    connections.to_csv(path + 'connections.csv', index=False)

def preprocess():
    ''' 
    Preprocessing of the data:
        1. Fix target of steel_plate_fault base table (1's and 2's -> 0's and 1's)
        2. Kidney disease has 2 misclassified values in target ('ckd\t' -> 'ckd')
        3. Rename the columns of the candidate tables and the keys in connections.csv files.
    '''
    steel_plate_fault = pd.read_csv('experiment_data/steel-plate-fault/steel_plate_fault.csv')
    steel_plate_fault['Class'] = steel_plate_fault['Class'] - 1
    steel_plate_fault.to_csv('experiment_data/steel-plate-fault/steel_plate_fault.csv', index=False)

    base_table = pd.read_csv('./experiment_data/kidney-disease/kidney_disease.csv')
    base_table['classification'] = base_table['classification'].replace(['ckd\t'], 'ckd')
    base_table.to_csv('experiment_data/kidney-disease/kidney_disease.csv', index=False)

    for path, target in datasets.items():
        connections = pd.read_csv(path + 'connections.csv')
        rename(path, connections)

def main():
    preprocess()
    
if __name__ == "__main__":
    main()