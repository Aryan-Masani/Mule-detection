import pandas as pd 

df = pd.read_csv('findataset.csv')

res = df[df['F3924'] == 1][['ID', 'F3891']]

print(res)