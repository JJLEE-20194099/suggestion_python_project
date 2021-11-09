import pandas as pd;

user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./data/ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')

n_users = users.shape[0]

print ('Number of users: ', n_users)
