import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, mean_squared_error, r2_score

def print_dataset_overview(dataset: pd.DataFrame, dataset_name: str):
  print()
  print('============================')
  print(dataset_name)
  print('============================')
  print('Attributes')
  print(dataset.columns)
  print()
  print('Number of rows')
  print(dataset.shape[0])
  print()
  print('Number of columns')
  print(dataset.shape[1])
  print()
  print('First 5 rows')
  print(dataset.head())
  print()
  print('Info')
  print(dataset.info())
  print()
  print('Describe')
  print(dataset.describe())

def plot_boxplot(dataset: pd.DataFrame, column: str):
    """
    Rysuje boxplot dla jednej kolumny w podanym DataFrame.
    """
    plt.figure(figsize=(6, 6))
    sns.set(style="whitegrid")

    sns.boxplot(y=dataset[column].dropna())
    plt.yscale('log')
    plt.title(f'Boxplot: {column}')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()

def plot_scatterplot(dataset: pd.DataFrame, x_col: str, y_col: str, y_log = False):
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")

    sns.scatterplot(data=dataset, x=x_col, y=y_col)
    if y_log:
        plt.yscale('log')
    plt.title(f'Scatterplot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()

def remove_outliers_3s(dataset: pd.DataFrame, x_col: str):
    p99 = dataset[x_col].quantile(0.997)
    print(p99)
    filtered_data = dataset[
        (dataset[x_col] <= p99)
    ]

    return filtered_data

def count_nans(dataset: pd.DataFrame):
  for column in dataset.columns:
    print(f'{column}: {dataset[column].isna().sum()}')

def draw_histogram(data, num_bins=12, title='Histogram', xlabel='Values', ylabel='Frequency'):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=num_bins, edgecolor='black', color='skyblue', log=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def print_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R-squared: {r2}")