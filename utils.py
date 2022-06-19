def drop_ids(columns):
    ids = ['id', 'Id', 'index']
    drop_cols = []
    for col in ids:
        drop_cols = drop_cols + list(filter(lambda x: col in x, columns))
    return drop_cols