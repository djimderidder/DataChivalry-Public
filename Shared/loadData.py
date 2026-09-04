import numpy as np
import pandas as pd
import os
import re

class FLUOstarData:
    def __init__(self, FileFolder, FileName):
        self.FileFolder = FileFolder
        self.FileName = FileName
        self.data = None
        self.dict = None
        self.load_data()
        self.gains = self.data['Gain'].unique()

    def load_data(self):
        path = os.path.join(self.FileFolder, self.FileName)
        N = 10
        for i in range(N+1):  # Assuming the header can be within the first 8 lines
            try:
                dataTemp = pd.read_csv(path, header=N-i, sep=",")
                if dataTemp.iloc[0, 0] == 'Well':
                    # Transpose the dataframe
                    dataTemp_T = dataTemp.transpose()
                    # Create the dictionary from the first and second row (excluding the first and second column)
                    self.dict = {dataTemp_T.iloc[0, i]: dataTemp_T.iloc[1, i] for i in range(2, dataTemp_T.shape[1])}
                    # Rename the columns to the first row
                    dataTemp_T.columns = dataTemp_T.iloc[0]
                    # Rename the Time [s] column to the second row
                    dataTemp_T.rename(columns={dataTemp_T.columns[1]: dataTemp_T.iloc[1, 1]}, inplace=True)
                    # Check if last row is not empty
                    if pd.isna(dataTemp_T['Well'].iloc[-1]):
                        dataTemp_T = dataTemp_T[:-1]
                    # Check if last column is not empty
                    if pd.isna(dataTemp_T.iloc[0, -1]):
                        dataTemp_T = dataTemp_T.iloc[:, :-1]
                    # Make a gain column
                    dataTemp_T.rename(columns={dataTemp_T.columns[0]: 'Gain'}, inplace=True)
                    dataTemp_T['Gain'] = dataTemp_T['Gain'].str[-2]
                    # Create a new dataframe removing the first two rows
                    self.data = dataTemp_T.iloc[2:, :].reset_index(drop=True)
                    # Convert the data to floats
                    self.data.replace('overflow', np.nan, inplace=True)
                    self.data = self.data.astype(float)
                    
                    if 'Time' not in self.data.columns[1]:
                        print('Time is not the second column')
                    break
            except Exception as e:
                print(f"Error reading file with header={N-i}: {e}")
        if self.data is None:
            print('Cannot read file')
        else:
            print('File read successfully')

    def set_gain(self, Gain):
        if len(self.gains) > 1:
            if Gain in self.gains:
                gaini = self.data['Gain'] == Gain
                self.data = self.data[gaini]
                self.gains = self.data['Gain'].unique()
            else:
                print('Gain not in data')
        else:
            print('Data already have one gain')
