import pandas as pd

# Filepath to the CSV file
csv_filepath = '/home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_additives/3_Make_New_Data_Predictor/ABC/cycle0.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_filepath)

# Extract the SMILES column
smiles_column = 'SMILES'  # Replace with the actual column name if different
mordred_column = 'ATSC5v'
mordred_df = df[mordred_column]
smiles_series = df[smiles_column]

# Count the occurrences of each unique SMILES string
smiles_counts = smiles_series.value_counts()
mordred_counts = mordred_df.value_counts() 
# Print the counts of unique SMILES strings
print(mordred_counts)
print(smiles_counts)

"""
# Optionally, save the counts to a new CSV file
output_filepath = '/home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_additives/3_Make_New_Data_Predictor/ABC/unique_smiles_counts.csv'
smiles_counts.to_csv(output_filepath, header=['Count'])
"""