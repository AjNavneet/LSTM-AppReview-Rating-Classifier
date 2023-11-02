import pandas as pd

class Load_Data:

    def load_data(self):
        """
        Load the data from a CSV file and select specific columns.
        :return: Loaded data with 'content' and 'score' columns.
        """
        # Read the data from the CSV file
        data = pd.read_csv('Input/review_data (1).csv')
        
        # Select the 'content' and 'score' columns from the loaded data
        data = data[['content', 'score']
        
        return data
