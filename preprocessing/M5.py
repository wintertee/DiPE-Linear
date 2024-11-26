import pandas as pd

train = pd.read_csv('sales_train_validation.csv')
test = pd.read_csv('sales_test_validation.csv')

df = pd.merge(train,
              test,
              on=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
              how='inner')

df = df.drop(columns=['item_id', 'dept_id', 'state_id']).groupby(
    ['cat_id', 'store_id']).sum().reset_index()

df['cat_id_store_id'] = df['cat_id'] + '_' + df['store_id']

df = df.set_index('cat_id_store_id').drop(['cat_id', 'store_id'], axis=1).T

df.insert(0, 'date', range(1, len(df) + 1))

df.to_csv('m5.csv.gz', index=False)
