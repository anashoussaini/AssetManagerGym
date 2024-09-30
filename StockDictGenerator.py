import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import sys

class StockDictGenerator:
    """
    StockDataProcessor class provides methods to load acronyms from a CSV file, create a filtered DataFrame,
    filter stock tickers based on continuous months of data, print feature correlations, and save the results to a JSON file.
    """

    def __init__(self, csv_file, df, additional_columns, columns_to_drop=None):
        """
        Initializes the StockDataProcessor with the CSV file, DataFrame, additional columns, and columns to drop.

        Parameters:
        -----------
        csv_file : str
            Path to the CSV file containing the acronyms.
        df : pd.DataFrame
            The full dataset that includes stock ticker data and associated metrics.
        additional_columns : list
            List of additional columns to keep from the original DataFrame (e.g., 'stock_ticker', 'year', 'month').
        columns_to_drop : list, optional
            List of columns to drop from the original DataFrame.

        Raises:
        -------
        ValueError : if the csv_file or additional_columns are invalid.
        """
        if not csv_file or not isinstance(csv_file, str):
            raise ValueError("Invalid CSV file path provided.")
        if not isinstance(additional_columns, list):
            raise ValueError("additional_columns should be a list.")

        self.csv_file = csv_file
        self.df = df
        self.additional_columns = additional_columns
        self.columns_to_drop = columns_to_drop if columns_to_drop else []

        self.acronyms = self.load_acronyms_from_csv()
        self.acronyms_df = self.create_acronyms_dataframe()

    def load_acronyms_from_csv(self):
        """
        Loads unique acronyms from the provided CSV file.

        Returns:
        --------
        list:
            A list of unique acronyms found in the CSV file under the 'Acronym' column.

        Raises:
        -------
        FileNotFoundError : if the CSV file cannot be found.
        KeyError : if the 'Acronym' column is missing from the CSV file.
        """
        try:
            acronyms_df = pd.read_csv(self.csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {self.csv_file} does not exist.")

        if 'Acronym' not in acronyms_df.columns:
            raise KeyError("The CSV file must contain an 'Acronym' column.")

        acronyms = acronyms_df['Acronym'].unique()
        return acronyms

    def create_acronyms_dataframe(self):
        """
        Creates a new DataFrame that contains only the selected acronyms and additional columns,
        and drops any specified columns. Additionally, it drops features with more than 50% missing
        values and replaces remaining NA values with the average of their respective numeric columns.
        All float64 columns are converted to float32.

        Returns:
        --------
        pd.DataFrame:
            A filtered DataFrame with acronyms, additional columns, and without the dropped columns.

        Raises:
        -------
        KeyError : if any of the required columns are missing from the DataFrame.
        """
        # Columns to keep (acronyms and additional columns)
        columns_to_keep = list(self.acronyms) + self.additional_columns

        # Check if any required columns are missing
        missing_columns = [col for col in columns_to_keep if col not in self.df.columns]
        if missing_columns:
            raise KeyError(f"The following columns are missing from the DataFrame: {missing_columns}")

        # Filter DataFrame to keep only the required columns
        acronyms_df = self.df[columns_to_keep]

        # Drop specified columns from the filtered DataFrame if they exist
        acronyms_df = acronyms_df.drop(columns=[col for col in self.columns_to_drop if col in acronyms_df.columns])

        # Drop columns with more than 50% missing values
        threshold = 0.5 * len(acronyms_df)
        acronyms_df = acronyms_df.dropna(axis=1, thresh=threshold)

        # Replace remaining NA values with the average of their respective numeric columns
        numeric_cols = acronyms_df.select_dtypes(include=['number']).columns
        acronyms_df[numeric_cols] = acronyms_df[numeric_cols].fillna(acronyms_df[numeric_cols].mean())

        # Convert all float64 columns to float32
        float_cols = acronyms_df.select_dtypes(include=['float64']).columns
        acronyms_df[float_cols] = acronyms_df[float_cols].astype('float32')

        # Store the filtered DataFrame in the class attribute
        self.acronyms_df = acronyms_df
        return acronyms_df

    def standardize_columns_global(self, df, exclude_columns):
        """
        Standardize all columns in the dataframe except for the ones listed in `exclude_columns`.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to standardize.
        exclude_columns : list
            A list of columns to exclude from standardization.

        Returns:
        --------
        pandas.DataFrame:
            The standardized DataFrame (with excluded columns unchanged).
        """
        # Select columns to standardize (those not in the exclude list)
        columns_to_standardize = df.columns.difference(exclude_columns)

        # Apply standardization for each column
        for col in columns_to_standardize:
            mean_value = df[col].mean()
            std_value = df[col].std(ddof=0)  # Use population std

            # Avoid division by zero for constant columns
            if std_value != 0:
                df[col] = (df[col] - mean_value) / std_value
            else:
                df[col] = 0  # Set all values to 0 if there's no variation in the column

        return df

    def normalize_columns_global(self, df, exclude_columns):
        """
        Normalize all columns in the dataframe except for the ones listed in `exclude_columns`.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to normalize.
        exclude_columns : list
            A list of columns to exclude from normalization.

        Returns:
        --------
        pandas.DataFrame:
            The normalized DataFrame (with excluded columns unchanged).
        """
        # Select columns to normalize (those not in the exclude list)
        columns_to_normalize = df.columns.difference(exclude_columns)
        
        # Apply min-max normalization globally for each column
        for col in columns_to_normalize:
            min_value = df[col].min()
            max_value = df[col].max()
            print(max_value)
            
            # Avoid division by zero for constant columns
            if max_value - min_value != 0:
                df[col] = (df[col] - min_value) / (max_value - min_value)
            else:
                df[col] = 0  # Set all values to 0 if there's no variation in the column

        return df

    def get_prc_statistics(self):
        """
        Calculate and return statistics for the 'prc' column in the DataFrame.

        Returns:
        --------
        dict:
            A dictionary containing statistics for the 'prc' column.
        """
        # Check if 'prc' column exists
        if 'prc' not in self.acronyms_df.columns:
            raise ValueError("The DataFrame must contain the 'prc' column.")

        # Retrieve the corresponding stock ticker
        # Get the top 10 stock tickers by 'prc'
        top_10_tickers = self.acronyms_df[['stock_ticker', 'prc']].nlargest(100, 'prc')
        print(top_10_tickers)
        # Calculate statistics
        stats = {
            'mean': self.acronyms_df['prc'].mean(),
            'median': self.acronyms_df['prc'].median(),
            'std_dev': self.acronyms_df['prc'].std(ddof=0),  # Population standard deviation
            'min': self.acronyms_df['prc'].min(),
            'max': self.acronyms_df['prc'].max(),
            '25th_percentile': self.acronyms_df['prc'].quantile(0.25),
            '50th_percentile': self.acronyms_df['prc'].quantile(0.5),
            '75th_percentile': self.acronyms_df['prc'].quantile(0.75),
            '99th_percentile': self.acronyms_df['prc'].quantile(0.99),
            'count': self.acronyms_df['prc'].count()
        }

        return stats


    def filter_tickers_by_continuous_months(self, min_months=12):
        """
        Filters the DataFrame for stock tickers that have at least `min_months` continuous months of data.

        Parameters:
        -----------
        min_months : int, optional
            Minimum number of continuous months required (default is 12).

        Returns:
        --------
        dict:
            A dictionary of stock tickers with continuous month data.
        """
        # Ensure required columns are present
        required_columns = ['stock_ticker', 'year', 'month']
        if not all(col in self.acronyms_df.columns for col in required_columns):
            raise ValueError(f"The DataFrame must contain the following columns: {required_columns}")

        # Sort the dataframe by stock_ticker, year, and month
        self.acronyms_df = self.acronyms_df.sort_values(by=required_columns)

        # Normalize all columns except for 'stock_ticker', 'year', and 'month'
        self.acronyms_df = self.normalize_columns_global(self.acronyms_df, exclude_columns=required_columns)

        # Group by stock_ticker
        grouped = self.acronyms_df.groupby('stock_ticker')

        # Create a dictionary to hold the filtered tickers
        filtered_tickers = {}

        # Iterate over each group (each stock ticker)
        for ticker, group in grouped:
            group = group.reset_index(drop=True)
            group['date'] = pd.to_datetime(group[['year', 'month']].assign(day=1))
            group['month_diff'] = group['date'].diff().dt.days // 30
            group['month_diff'].fillna(1, inplace=True)

            current_streak = 0
            start_index = None
            for i, diff in enumerate(group['month_diff']):
                if diff == 1:
                    if current_streak == 0:
                        start_index = i
                    current_streak += 1
                else:
                    current_streak = 1
                    start_index = i

                if current_streak >= min_months:
                    for _, row in group.iloc[start_index:start_index + min_months].iterrows():
                        year = row['year']
                        month = row['month']
                        rest_of_columns = row.drop(['stock_ticker', 'year', 'month', 'date', 'month_diff']).to_dict()

                        if ticker not in filtered_tickers:
                            filtered_tickers[ticker] = {}
                        if year not in filtered_tickers[ticker]:
                            filtered_tickers[ticker][year] = {}

                        filtered_tickers[ticker][year][month] = rest_of_columns

                    break

        return filtered_tickers

    def print_high_correlation_pairs(self, threshold=0.75):
        """
        Prints all pairs of features with an absolute correlation higher than the threshold.

        Parameters:
        -----------
        threshold : float, optional
            The correlation threshold for printing feature pairs (default is 0.75).
        """
        # Filter only numeric columns
        numeric_df = self.acronyms_df.select_dtypes(include=['number'])

        if numeric_df.empty:
            print("No numeric columns found in the DataFrame to compute correlations.")
            return

        # Calculate the correlation matrix
        corr_matrix = numeric_df.corr().abs()

        # Find all pairs of features with correlation above the threshold
        high_corr_pairs = (corr_matrix.where(
            lambda x: (x > threshold) & (x != 1))  # Filter out self-correlations (diagonal values)
        ).stack().sort_values(ascending=False)

        # Print each high-correlation pair
        if high_corr_pairs.empty:
            print(f"No pairs of features with correlation above {threshold}")
        else:
            for (feature1, feature2), correlation in high_corr_pairs.items():
                print(f"{feature1} and {feature2} have a correlation of {correlation:.2f}")

    def save_dict_to_json(self,data,file_path):
        """
        Saves the filtered ticker data to a JSON file in a human-readable format.

        Parameters:
        -----------
        data : dict
            The dictionary to save.
        file_path : str
            The file path where the JSON data will be saved.
        """
        if not isinstance(file_path, str):
            raise ValueError("Invalid file path provided.")

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)  # indent=4 makes the file human-readable

    def plot_correlation_heatmap(self, output_pdf_path):
        """
        Plots a correlation heatmap of the acronyms data and saves it as a PDF, excluding additional columns
        and handling dropped columns.

        Parameters:
        -----------
        output_pdf_path : str
            The file path where the heatmap PDF will be saved.
        """
        # Ensure we are working with the correct remaining acronym columns in acronyms_df
        remaining_acronyms = [col for col in self.acronyms if col in self.acronyms_df.columns]

        # Filter the DataFrame to only include the remaining acronym columns
        acronyms_only_df = self.acronyms_df[remaining_acronyms]

        # Compute the correlation matrix for acronym columns only
        correlation_matrix = acronyms_only_df.corr()

        # Increase the figure size to avoid truncation
        plt.figure(figsize=(14, 12))  # Adjust size as needed

        # Plot heatmap using seaborn with larger size and adjust other settings
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5,
                    cbar_kws={'shrink': 0.8})  # Shrink color bar for better fit

        # Add title to the heatmap
        plt.title('Correlation Heatmap of Acronym Metrics', fontsize=16)

        # Adjust layout to fit everything
        plt.tight_layout()

        # Save the plot as a PDF
        plt.savefig(output_pdf_path, format='pdf')

        # Close the plot to free memory
        plt.close()

# Example usage:
if __name__ == "__main__":
    # Load your stock dataset
    df = pd.read_csv('hackathon_sample_v2.csv')  # Your actual DataFrame source
    csv_file = 'metrics_acronyms.csv'  # CSV file containing acronyms
    additional_columns = ['stock_ticker', 'year', 'month']
    columns_to_drop = ['ncoa_gr1a', 'be_gr1a', 'nfna_gr1a','ncol_gr1a', 'ebitda_mev','debt_me']
    # Instantiate the class
    processor = StockDictGenerator(csv_file, df, additional_columns, columns_to_drop)

    # Print pairs of features with correlation above 0.75
    #processor.print_high_correlation_pairs(threshold=0.75)

    #processor.plot_correlation_heatmap('heatmap_output.pdf')

    feature_dict = processor.filter_tickers_by_continuous_months()

    processor.save_dict_to_json(feature_dict, 'feature_dict.json')

    #print(processor.get_prc_statistics())