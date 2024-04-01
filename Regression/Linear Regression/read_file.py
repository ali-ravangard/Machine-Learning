import pandas as pd 
import os 

def read_data(file_name, format='csv'):

    cwd_py = os.path.dirname(os.path.realpath(__file__))  # get the current directory 
    os.chdir(cwd_py) # change directory to current 

    if format == 'csv': # read file if its format is csv
        df = pd.read_csv(file_name, sep=',', header=None)
        df = df.to_numpy() # change dataframe to numpy array
    elif format == 'txt': # read file if its format is txt (the seperator is tab space)
        df = pd.read_csv(file_name, sep='\t', header=None)
        df = df.to_numpy()

    return df 


