import pandas as pd

# Load the CSV file
df = pd.read_csv('mdataset.csv')

# Define the columns you want to keep
columns_to_keep = ['ID','F115',
                    'F321',
                    'F527',
                    'F531',
                    'F670',
                    'F1692',
                    'F2082',
                    'F2122',
                    'F2582',
                    'F2678',
                    'F2737',
                    'F2956',
                    'F3043',
                    'F3836',
                    'F3887',
                    'F3889',
                    'F3891',
                    'F3894',
                    'F3924']

# Filter the dataframe
df_filtered = df[columns_to_keep]

# Save the result to a new file
df_filtered.to_csv('findataset.csv', index=False)