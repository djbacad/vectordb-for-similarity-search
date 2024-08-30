import pandas as pd

# Load the dataset
df = pd.read_csv('./datasets/questions.csv')

# Sample 5000 rows
df_sampled = df.sample(n=10000, random_state=43)

# Save the sampled data to a new CSV file
df_sampled.to_csv('datasets/questions_sampled.csv', index=False)