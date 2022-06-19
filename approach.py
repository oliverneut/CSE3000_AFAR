import pandas as pd
import numpy as np
from datetime import datetime
from utils import drop_ids
from models import train_SVM, train_cart, train_randomforest, train_xgboost
from sklearn.model_selection import train_test_split
from ITMO_FS import su_measure, gini_index, information_gain
import warnings
warnings.filterwarnings('ignore')


datasets = {
    'experiment_data/football/': 'win',
    'experiment_data/kidney-disease/': 'classification',
    'experiment_data/steel-plate-fault/': 'Class',
    'experiment_data/titanic/': 'Survived'
}

def join_paths(path, connections, target):
    '''
    Stratify joins the different tables to the path using the connections files and returns it
    '''
    base = connections['from_table'][0]
    base_table = pd.read_csv(path + base)

    sample = train_test_split(base_table.drop(target, axis=1), base_table[target], test_size=0.5, stratify=base_table[target], random_state=1)
    idxs = sample[0].index
    res = {base:base_table.iloc[idxs]}

    for index, row in connections.iterrows():
        pk, fk = row['from_key'], row['to_key']
        from_table, to_table = row['from_table'], row['to_table']
        new_table = pd.read_csv(path + to_table)

        res[to_table] = pd.merge(res[from_table], new_table, how='left', left_on=pk, right_on=fk)

    return res

def pearson_score(df, base_cols, cand_cols, target):
    '''
    Computes Pearson correlation of candidate table and target, 
    and for each feature of candidate table, the Pearson correlation with base table is computed.
    We store the max feature-target correlation times the max feature-feature non-correlation. 
    We return the max value of this
    '''
    t_corr = df[cand_cols].copy()
    t_corr[target] = df[target]
    CT = abs(t_corr.corr(method='pearson')[target]).sort_values(ascending=False)[1:]
    
    for c_col in cand_cols:
        b_corr = df[base_cols].copy()
        b_corr[c_col] = df[c_col]
        CB = max(abs(b_corr.corr(method='pearson')[c_col]).sort_values(ascending=False)[1:])
        CT[c_col] = CT[c_col] * (1 - CB)
    
    return max(CT)

def symetrical_uncertainty_score(df, base_cols, cand_cols, target):
    '''
    Computes symmetrical uncertainty of candidate table and target
    returns the max value
    '''
    t_corr = np.array(df[cand_cols].copy())
    CT = su_measure(t_corr, np.array(df[target]))    
    return max(CT)

def infogain_score(df, base_cols, cand_cols, target):
    '''
    Computes information gain of candidate table and target
    returns the max value
    '''
    t_corr = np.array(df[cand_cols].copy())
    CT = information_gain(t_corr, np.array(df[target]))        
    return max(CT)

def gini_score(df, base_cols, cand_cols, target):
    '''
    Computes gini index of candidate table and target
    returns 1 - the lowest value
    '''
    t_corr = np.array(df[cand_cols].copy())
    CT = gini_index(t_corr, np.array(df[target]))
    return 1 - min(CT)

def score(method, c_table, b_columns, c_columns, target):
    if method == 'pearson':
        return pearson_score(c_table, b_columns, c_columns, target)
    elif method == 'infogain':
        return infogain_score(c_table, b_columns, c_columns, target)
    elif method == 'gini':
        return gini_score(c_table, b_columns, c_columns, target)
    elif method == 'su':
        return symetrical_uncertainty_score(c_table, b_columns, c_columns, target)

def get_full_join(connections, path, base, best_cand):
    '''
    joins a candidate table with the base table
    '''
    res = pd.read_csv(path + best_cand)
    flag = True
    while flag:
        for index, row in connections.iterrows():
            if row['to_table'] == best_cand:
                pk, fk = row['from_key'], row['to_key']
                from_table = row['from_table']
                new_table = pd.read_csv(path + from_table)
                res = pd.merge(new_table, res, how='left', left_on=pk, right_on=fk)
                if from_table == base:
                    flag = False
                    break
                best_cand = from_table
    return res

def rank_1(path, paths, b_columns, target, method='pearson'):
    '''
    Rank 1: uses pearson score
    returns the ranked candidate tables
    '''
    Q = []
    for candidate in paths:
        c_table = paths[candidate]
        c_table = c_table.apply(lambda x: pd.factorize(x)[0])
        c_columns = pd.read_csv(path + candidate).columns
        c_columns = c_columns.drop(drop_ids(c_columns))

        score_pearson = score(method, c_table, b_columns, c_columns, target)
        Q.append([candidate, score_pearson, c_columns])

    Q = np.array(Q)
    Q_sorted = Q[Q[:,1].argsort()[::-1]]
    return Q_sorted

def rank_2(path, paths, b_columns, target):
    '''
    Rank 2: uses Pearson score, information gain, Gini index, unique values formula, 
    Normalizes these scores, sums these scores and returns the ranked candidate tables
    '''
    Q = []
    scores = []
    for candidate in paths:
        c_table = paths[candidate]
        c_table = c_table.apply(lambda x: pd.factorize(x)[0])
        c_columns = pd.read_csv(path + candidate).columns
        c_columns = c_columns.drop(drop_ids(c_columns))

        score_1 = np.max([(1/n) for n in c_table[c_columns].nunique()])   # unique values formula
        score_2 = score('pearson', c_table, b_columns, c_columns, target)
        score_3 = score('infogain', c_table, b_columns, c_columns, target)
        score_4 = score('gini', c_table, b_columns, c_columns, target)
        scores.append([score_1, score_2, score_3, score_4])
        Q.append([candidate, 0, c_columns])

    scores = np.array(scores)
    scores_normalized = scores / scores.max(axis=0)
    score_rank = scores_normalized.sum(axis=1)

    Q = np.array(Q)
    Q[:,1] = score_rank

    Q_sorted = Q[Q[:,1].argsort()[::-1]]

    return Q_sorted

def best_joinpath_approach():
    results = []

    for path, target in datasets.items():
        start_time = datetime.now()
        connections = pd.read_csv(path + 'connections.csv')
        paths = join_paths(path, connections, target)

        base = connections['from_table'][0]
        b_columns = paths.pop(base).columns
        b_columns = b_columns.drop(drop_ids(b_columns))
        b_columns = b_columns.drop(target)
        
        Q_sorted = rank_1(path, paths, b_columns, target, 'su')
        best_path = Q_sorted[0,0]
        print(best_path)
        
        df = get_full_join(connections, path, base, best_path)
        df.drop(drop_ids(df.columns), axis=1, inplace=True)

        path_columns = list(b_columns) + list(Q_sorted[0][2]) + list([target])
        df = df[path_columns]

        df = df.apply(lambda x: pd.factorize(x)[0])

        X, y = df.drop(target, axis=1), df[target]

        acc, depth = train_xgboost(X, y)
        # acc, depth = train_SVM(X, y), 0
        # acc, depth = train_randomforest(X, y)
        # acc, depth = train_cart(X, y)
        end_time = datetime.now()

        time = end_time - start_time
        res = { 
            'dataset': base[:-4],
            'join_path': best_path,
            'accuracy': acc,
            'depth': depth,
            'runtime': time.total_seconds(),
            'accuracy_order': Q_sorted[:,0]
        }
        results.append(res)

    pd.DataFrame(results).to_csv('results/approach_CART_su.csv', index=False)

def join_table(connections, path, table, c_table):
    print(c_table)
    right = pd.read_csv(path + c_table)
    flag = True
    while flag:
        for index, row in connections.iterrows():
            if row['to_table'] == c_table:
                pk, fk = row['from_key'], row['to_key']
                if fk in table.columns:
                    flag = False
                    break
                if pk in table.columns:
                    table = pd.merge(table, right, how='left', left_on=pk, right_on=fk)
                    flag = False
                    break
                else:
                    c_table = row['from_table']
                    right = pd.merge(pd.read_csv(path + row['from_table']), right, how='left', left_on=pk, right_on=fk)
    return table

def join_oneByOne():
    results = []
    for path, target in datasets.items():
        
        start_time = datetime.now()
        connections = pd.read_csv(path + 'connections.csv')
        paths = join_paths(path, connections, target)

        base = connections['from_table'][0]
        b_table = pd.read_csv(path + base)
        b_columns = paths.pop(base).columns
        b_columns = b_columns.drop(drop_ids(b_columns))
        b_columns = b_columns.drop(target)
        
        Q_sorted = rank_2(path, paths, b_columns, target)
        best_path = Q_sorted[0,0]
        print(best_path)

        path_columns = list(b_columns) + list([target])
        for row in Q_sorted:
            b_table = join_table(connections, path, b_table, row[0])            
            # print(b_table.columns)
            path_columns = path_columns + list(row[2])
            print(path_columns)
            table = b_table[path_columns]
            table.drop(drop_ids(table.columns), axis=1, inplace=True)
    
            table = table.apply(lambda x: pd.factorize(x)[0])

            X, y = table.drop(target, axis=1), table[target]
            acc, depth = train_xgboost(X, y)
            
            stop_time = datetime.now()
            time = stop_time - start_time
            res = {
                'dataset': base[:-4],
                'join_path': row[0],
                'accuracy': acc,
                'depth': depth,
                'runtime': time.total_seconds()
            }
            results.append(res) 
    pd.DataFrame(results).to_csv('results/join_approach_1.csv', index=False)

def verify_order():

    for path, target in datasets.items():
        Q = []
        connections = pd.read_csv(path + 'connections.csv')
        base = connections['from_table'][0]
        base_table = pd.read_csv(path + base)
        print(base)

        paths = {base:base_table}

        for index, row in connections.iterrows():
            pk, fk = row['from_key'], row['to_key']
            from_table, to_table = row['from_table'], row['to_table']
            new_table = pd.read_csv(path + to_table)

            paths[to_table] = pd.merge(paths[from_table], new_table, how='left', left_on=pk, right_on=fk)

        b_columns = paths.pop(base).columns
        b_columns = b_columns.drop(drop_ids(b_columns))

        for candidate in paths:
            print(candidate, end = ', ')
            c_table = paths[candidate]
            c_columns = pd.read_csv(path + candidate).columns
            c_columns = c_columns.drop(drop_ids(c_columns))
            path_columns = list(b_columns) + list(c_columns)
            c_table = c_table[path_columns]
            c_table = c_table.apply(lambda x: pd.factorize(x)[0])
            
            X, y = c_table.drop(target, axis=1), c_table[target]
            acc, depth = train_xgboost(X, y)
            Q.append([candidate, acc])
            Q.sort(key=lambda row: row[1], reverse=True)

        res = pd.DataFrame(Q, columns=['candidate', 'accuracy'])
        pd.DataFrame(res).to_csv('results/accuracy_order_'+ base, index=False)

if __name__ == '__main__':
    join_oneByOne()