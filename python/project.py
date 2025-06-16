import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

USA_REAL_ESTATE_DATASET_PATH = 'data/usa-real-estate-dataset.csv'
USACITIES_PATH = 'data/uscities.csv'
ured = pd.read_csv(USA_REAL_ESTATE_DATASET_PATH, dtype={'zip_code': str})
uscities = pd.read_csv(USACITIES_PATH)

def print_dataset_overview(dataset, dataset_name):
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

print_dataset_overview(ured, 'USA Real Estate Dataset')

ured_filtered = ured.drop(columns=['brokered_by', 'street', 'status'])

print_dataset_overview(ured_filtered, 'USA Real Estate Dataset - filtered')

target = ured_filtered['price']
ured_filtered.describe()

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

plot_boxplot(ured_filtered, 'bed')
plot_boxplot(ured_filtered, 'bath')
plot_boxplot(ured_filtered, 'acre_lot')
plot_boxplot(ured_filtered, 'house_size')
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
plot_scatterplot(ured_filtered, 'bed', 'bath')
ured_filtered = ured_filtered[
    ((ured_filtered['bath'] < 2 * ured_filtered['bed']) |
     (ured_filtered['bed'].isna()) |
      (ured_filtered['bath'].isna())) & (ured_filtered['bed'] < 100)
        ]
plot_scatterplot(ured_filtered, 'bed', 'bath')
ured_filtered = ured_filtered[ured_filtered['acre_lot'] != 0]
ured_filtered = ured_filtered[ured_filtered['house_size'] != 0]
plot_scatterplot(ured_filtered, 'acre_lot', 'house_size', y_log=True)
def remove_outliers(dataset: pd.DataFrame, x_col: str):
    p99 = dataset[x_col].quantile(0.997)
    print(p99)
    filtered_data = dataset[
        (dataset[x_col] <= p99)
    ]

    return filtered_data
ured_outliers_fixed = remove_outliers(ured_filtered, 'house_size')
ured_filtered = ured_outliers_fixed
plot_scatterplot(ured_filtered, 'acre_lot', 'house_size')
ured_filter_date = ured_filtered
ured_filter_date = ured_filter_date[
    (ured_filter_date['prev_sold_date'].isnull()) |
    (ured_filter_date['prev_sold_date']
     .str.slice(0, 4)
     .apply(pd.to_numeric, errors='coerce')
     .between(1900, 2025))
]
ured_filtered = ured_filter_date
def count_nans(dataset):
  for column in dataset.columns:
    print(f'{column}: {dataset[column].isna().sum()}')
count_nans(ured_filtered)
ured_filtered = ured_filtered.dropna(subset=['price', 'city', 'state', 'zip_code'])
ured_filtered = ured_filtered.fillna({'bed': ured_filtered['bed'].mode()[0],
                                      'bath': ured_filtered['bath'].mode()[0],
                                      'acre_lot': ured_filtered['acre_lot'].median(),
                                      'house_size': ured_filtered['house_size'].median()})
ured_filtered['sold_before'] = 0
ured_filtered.loc[ured_filtered['prev_sold_date'].notna(), 'sold_before'] = 1
ured_filtered['years_since_sold'] = 0
ured_filtered.loc[ured_filtered['prev_sold_date'].notna(), 'years_since_sold'] = 2024 - pd.to_datetime(ured_filtered.loc[ured_filtered['prev_sold_date'].notna(), 'prev_sold_date']).dt.year
ured_filtered = ured_filtered.drop(columns=['prev_sold_date'])
ured_filtered.head()
count_nans(ured_filtered)
def draw_histogram(data, num_bins=12, title='Histogram', xlabel='Values', ylabel='Frequency'):
    """
    Draws a histogram from the provided data.

    Parameters:
    - data: list or array of numeric values
    - num_bins: number of bins for the histogram (default is 10)
    - title: title of the plot
    - xlabel: label for the X-axis
    - ylabel: label for the Y-axis
    """
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=num_bins, edgecolor='black', color='skyblue', log=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
draw_histogram(ured_filtered['bed'], title='Bed frequency', xlabel='Bed', ylabel='Frequency')
draw_histogram(ured_filtered['bath'], title='bath frequency', xlabel='bath', ylabel='Frequency')
draw_histogram(ured_filtered['acre_lot'], title='acre_lot frequency', xlabel='acre_lot', ylabel='Frequency')
draw_histogram(ured_filtered['house_size'], title='house_size frequency', xlabel='house_size', ylabel='Frequency')
draw_histogram(ured_filtered['years_since_sold'], title='years_since_sold frequency', xlabel='years_since_sold', ylabel='Frequency')
import numpy as np
ured_scaled = ured_filtered.copy()
ured_scaled['house_size'] = (ured_filtered['house_size'] - ured_filtered['house_size'].min()) / (ured_filtered['house_size'].max() - ured_filtered['house_size'].min())
ured_scaled['acre_lot'] = (ured_filtered['acre_lot'] - ured_filtered['acre_lot'].min()) / (ured_filtered['acre_lot'].max() - ured_filtered['acre_lot'].min())
draw_histogram(ured_scaled['house_size'], title='house_size frequency', xlabel='house_size', ylabel='Frequency')
draw_histogram(ured_scaled['acre_lot'], title='acre_lot frequency', xlabel='acre_lot', ylabel='Frequency')
from scipy import stats

ured_normalized = ured_scaled.copy()

ured_normalized['house_size'], lambda_house_size = stats.boxcox(ured_scaled['house_size'] + 1)

draw_histogram(ured_normalized['house_size'], title='house_size frequency', xlabel='house_size', ylabel='Frequency')
print_dataset_overview(uscities, 'USA cities')
import pandas as pd

city_state_population = {}
mean_pop = uscities['population'].mean()
for index, row in uscities.iterrows():
    city_state_population[(row['city_ascii'], row['state_name'])] = row['population']

def get_population(city, state):
    if (city, state) in city_state_population:
        return city_state_population[(city, state)]
    else:
        return mean_pop

ured_normalized['city_population'] = ured_normalized.apply(lambda row: get_population(row['city'], row['state']), axis=1)

print_dataset_overview(ured_normalized, 'USA cities')
ured_normalized['city_population'] = (ured_normalized['city_population'] - ured_normalized['city_population'].min()) / (ured_normalized['city_population'].max() - ured_normalized['city_population'].min())
ured_2zipcode = ured_normalized.copy()
#take the first 2 digits of the zip_code
ured_2zipcode['zip_code'] = ured_2zipcode['zip_code'].str.slice(0, 2)
ured_2zipcode.to_csv('usa-real-estate-dataset-normalized-zipcodes-f2cat.csv', index=False)
ured_2zipcode['zip_code'] = ured_2zipcode['zip_code'].astype(float)
ured_2zipcode.to_csv('usa-real-estate-dataset-normalized-zipcodes-f2num.csv', index=False)
ured_normalized['zip_code'] = ured_normalized['zip_code'].astype(float)
ured_normalized.to_csv('usa-real-estate-dataset-normalized.csv', index=False)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample


ured_final= pd.read_csv('usa-real-estate-dataset-normalized.csv')
ured_final=ured_normalized.copy()
ured_final['state'] = ured_final['state'].astype('category')
ured_final['city'] = ured_final['city'].astype('category')
ured_final['zip_code'] = ured_final['zip_code'].astype('category')
X = ured_final.drop(columns=['price'])
y = ured_final['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)
combined_sample = pd.concat([X_train, y_train], axis=1).sample(n=10000, random_state=2137)
X_small = combined_sample.drop(columns=['price'])
X_small['state'] = X_small['state'].astype('category')
X_small['city'] = X_small['city'].astype('category')
X_small['zip_code'] = X_small['zip_code'].astype('category')
y_small = combined_sample['price']
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5],
    'min_child_weight': [1, 3, 5]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=1697, enable_categorical=True)

xgb_random = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=1697, n_jobs=-1)
xgb_random.fit(X_small, y_small)

print(xgb_random.best_params_)

best_random = xgb_random.best_estimator_
best_random.fit(X_train, y_train)

y_pred = xgb_random.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ured_normalized = pd.read_csv('usa-real-estate-dataset-normalized.csv')

categorical_cols = ['state', 'city']

top_cities = (
    ured_normalized['city']
    .value_counts()
    .nlargest(200)
    .index
)

ured_normalized['city'] = ured_normalized['city'].where(ured_normalized['city'].isin(top_cities), other='__other__')

enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
encoded_features = enc.fit_transform(ured_normalized[categorical_cols])

encoded_col_names = enc.get_feature_names_out(categorical_cols)

encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_features, columns=encoded_col_names)

ured_encoded = pd.concat([ured_normalized.drop(columns=categorical_cols), encoded_df], axis=1)

X_encoded = ured_encoded.drop(columns=['price'])
y = ured_encoded['price']

from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X_encoded, y)
feature_scores = pd.Series(selector.scores_, index=X_encoded.columns)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(feature_scores.sort_values(ascending=False))

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
ured_normalized = pd.read_csv('usa-real-estate-dataset-normalized.csv')

top_cities = ured_normalized['city'].value_counts().nlargest(200).index
ured_normalized['city'] = ured_normalized['city'].where(ured_normalized['city'].isin(top_cities), '__other__')

ured_dummies = pd.get_dummies(ured_normalized, columns=['city', 'state'], drop_first=True)

zip_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ured_dummies[['zip_code']] = zip_encoder.fit_transform(ured_dummies[['zip_code']])
ured_dummies['zip_code'] = ured_dummies['zip_code'].astype('category')
q_high = ured_dummies['price'].quantile(0.99)
ured_dummies = ured_dummies[ured_dummies['price'] < q_high]
X = ured_dummies.drop('price', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = ured_dummies['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=2137, enable_categorical=True, subsample=0.8, n_estimators=1000, min_child_weight=3, max_depth=7, learning_rate=0.01, gamma=0, colsample_bytree=0.5)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

xgb_model.save_model("models/xgb_model_200c_50s_zipcode_cat.json")

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

ured_normalized = pd.read_csv('usa-real-estate-dataset-normalized-zipcodes-f2cat.csv')

top_cities = ured_normalized['city'].value_counts().nlargest(200).index
ured_normalized['city'] = ured_normalized['city'].where(ured_normalized['city'].isin(top_cities), '__other__')

ured_dummies = pd.get_dummies(ured_normalized, columns=['city', 'state'], drop_first=True)

zip_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ured_dummies[['zip_code']] = zip_encoder.fit_transform(ured_dummies[['zip_code']])
ured_dummies['zip_code'] = ured_dummies['zip_code'].astype('category')
q_high = ured_dummies['price'].quantile(0.99)
ured_dummies = ured_dummies[ured_dummies['price'] < q_high]

X = ured_dummies.drop('price', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = ured_dummies['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=2137, enable_categorical=True, subsample=0.8, n_estimators=1000, min_child_weight=3, max_depth=7, learning_rate=0.01, gamma=0, colsample_bytree=0.5)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

xgb_model.save_model("models/xgb_model_200c_50s_zipcode_f2cat.json")

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler

ured_normalized = pd.read_csv('usa-real-estate-dataset-normalized-zipcodes-f2num.csv')

top_cities = ured_normalized['city'].value_counts().nlargest(200).index
ured_normalized['city'] = ured_normalized['city'].where(ured_normalized['city'].isin(top_cities), '__other__')

ured_dummies = pd.get_dummies(ured_normalized, columns=['city', 'state'], drop_first=True)

# zip_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# ured_dummies[['zip_code']] = zip_encoder.fit_transform(ured_dummies[['zip_code']])
# ured_dummies['zip_code'] = ured_dummies['zip_code'].astype('category')
q_high = ured_dummies['price'].quantile(0.99)
ured_dummies = ured_dummies[ured_dummies['price'] < q_high]

X = ured_dummies.drop('price', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = ured_dummies['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=2137, enable_categorical=True, subsample=0.8, n_estimators=1000, min_child_weight=3, max_depth=7, learning_rate=0.01, gamma=0, colsample_bytree=0.5)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

xgb_model.save_model("models/xgb_model_200c_50s_zipcode_f2num.json")

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler

ured_normalized = pd.read_csv('usa-real-estate-dataset-normalized-zipcodes-f2cat.csv')
ured_normalized = ured_normalized.drop(columns=['state', 'city'])
# top_cities = ured_normalized['city'].value_counts().nlargest(200).index
# ured_normalized['city'] = ured_normalized['city'].where(ured_normalized['city'].isin(top_cities), '__other__')

# ured_dummies = pd.get_dummies(ured_normalized, columns=['city', 'state'], drop_first=True)
ured_dummies = ured_normalized.copy()
zip_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ured_dummies[['zip_code']] = zip_encoder.fit_transform(ured_dummies[['zip_code']])
ured_dummies['zip_code'] = ured_dummies['zip_code'].astype('category')
q_high = ured_dummies['price'].quantile(0.99)
ured_dummies = ured_dummies[ured_dummies['price'] < q_high]
X = ured_dummies.drop('price', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = ured_dummies['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=2137, enable_categorical=True, subsample=0.8, n_estimators=1000, min_child_weight=3, max_depth=7, learning_rate=0.01, gamma=0, colsample_bytree=0.5)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

xgb_model.save_model("models/xgb_model_zipcode_f2cat.json")

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler

ured_normalized = pd.read_csv('usa-real-estate-dataset-normalized-zipcodes-f2num.csv')
ured_normalized = ured_normalized.drop(columns=['state', 'city'])
# top_cities = ured_normalized['city'].value_counts().nlargest(200).index
# ured_normalized['city'] = ured_normalized['city'].where(ured_normalized['city'].isin(top_cities), '__other__')

# ured_dummies = pd.get_dummies(ured_normalized, columns=['city', 'state'], drop_first=True)

# zip_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# ured_dummies[['zip_code']] = zip_encoder.fit_transform(ured_dummies[['zip_code']])
# ured_dummies['zip_code'] = ured_dummies['zip_code'].astype('category')
ured_dummies = ured_normalized.copy()
q_high = ured_dummies['price'].quantile(0.99)
ured_dummies = ured_dummies[ured_dummies['price'] < q_high]

X = ured_dummies.drop('price', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = ured_dummies['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=2137, enable_categorical=True, subsample=0.8, n_estimators=1000, min_child_weight=3, max_depth=7, learning_rate=0.01, gamma=0, colsample_bytree=0.5)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

xgb_model.save_model("models/xgb_model_zipcode_f2num.json")

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler

ured_normalized = pd.read_csv('usa-real-estate-dataset-normalized.csv')
ured_normalized = ured_normalized.drop(columns=['state', 'city'])
# top_cities = ured_normalized['city'].value_counts().nlargest(200).index
# ured_normalized['city'] = ured_normalized['city'].where(ured_normalized['city'].isin(top_cities), '__other__')

# ured_dummies = pd.get_dummies(ured_normalized, columns=['city', 'state'], drop_first=True)
ured_dummies = ured_normalized.copy()
zip_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
ured_dummies[['zip_code']] = zip_encoder.fit_transform(ured_dummies[['zip_code']])
ured_dummies['zip_code'] = ured_dummies['zip_code'].astype('category')
q_high = ured_dummies['price'].quantile(0.99)
ured_dummies = ured_dummies[ured_dummies['price'] < q_high]

X = ured_dummies.drop('price', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = ured_dummies['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=2137, enable_categorical=True, subsample=0.8, n_estimators=1000, min_child_weight=3, max_depth=7, learning_rate=0.01, gamma=0, colsample_bytree=0.5)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

xgb_model.save_model("models/xgb_model_zipcode_cat.json")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt

def nn_model(model_name, X, y):

    os.makedirs("models", exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2137)
    # === Model Definition ===
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    # === Callbacks ===
    checkpoint_cb = ModelCheckpoint(
        filepath='models/best_model.keras',
        monitor='val_mae',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )

    early_stop_cb = EarlyStopping(
        monitor='val_mae',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )


    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[checkpoint_cb, early_stop_cb]
    )

    model.save(f"models/{model_name}.keras")

    # === Evaluate on test set ===
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")

    # === Predict ===
    y_pred_log = model.predict(X_test)

    # Revert log1p transformation
    y_pred_actual = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.legend()
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True)
    plt.show()

zip_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)


ured_normalized = pd.read_csv('usa-real-estate-dataset-normalized.csv')
top_cities = ured_normalized['city'].value_counts().nlargest(200).index
ured_normalized['city'] = ured_normalized['city'].where(ured_normalized['city'].isin(top_cities), '__other__')
q_high = ured_normalized['price'].quantile(0.99)
ured_normalized = ured_normalized[ured_normalized['price'] < q_high]

ured_f2cat = pd.read_csv('usa-real-estate-dataset-normalized-zipcodes-f2cat.csv')
top_cities = ured_f2cat['city'].value_counts().nlargest(200).index
ured_f2cat['city'] = ured_f2cat['city'].where(ured_f2cat['city'].isin(top_cities), '__other__')
q_high = ured_f2cat['price'].quantile(0.99)
ured_f2cat = ured_f2cat[ured_f2cat['price'] < q_high]

ured_f2num = pd.read_csv('usa-real-estate-dataset-normalized-zipcodes-f2num.csv')
top_cities = ured_f2num['city'].value_counts().nlargest(200).index
ured_f2num['city'] = ured_f2num['city'].where(ured_f2num['city'].isin(top_cities), '__other__')
q_high = ured_f2num['price'].quantile(0.99)
ured_f2num = ured_f2num[ured_f2num['price'] < q_high]


ured_full_zipcode_geo = ured_normalized.copy()
ured_full_zipcode_geo = pd.get_dummies(ured_full_zipcode_geo, columns=['city', 'state'], drop_first=True)
ured_full_zipcode_geo[['zip_code']] = zip_encoder.fit_transform(ured_full_zipcode_geo[['zip_code']])
ured_full_zipcode_geo['zip_code'] = ured_full_zipcode_geo['zip_code'].astype('category')

ured_full_zipcode_nogeo = ured_normalized.drop(columns=['city', 'state'])
ured_full_zipcode_nogeo[['zip_code']] = zip_encoder.fit_transform(ured_full_zipcode_nogeo[['zip_code']])
ured_full_zipcode_nogeo['zip_code'] = ured_full_zipcode_nogeo['zip_code'].astype('category')

ured_zipcode_f2cat_geo = ured_f2cat.copy()
ured_zipcode_f2cat_geo = pd.get_dummies(ured_zipcode_f2cat_geo, columns=['city', 'state'], drop_first=True)
ured_zipcode_f2cat_geo[['zip_code']] = zip_encoder.fit_transform(ured_zipcode_f2cat_geo[['zip_code']])
ured_zipcode_f2cat_geo['zip_code'] = ured_zipcode_f2cat_geo['zip_code'].astype('category')

ured_zipcode_f2cat_nogeo = ured_f2cat.drop(columns=['city', 'state'])
ured_zipcode_f2cat_nogeo[['zip_code']] = zip_encoder.fit_transform(ured_zipcode_f2cat_nogeo[['zip_code']])
ured_zipcode_f2cat_nogeo['zip_code'] = ured_zipcode_f2cat_nogeo['zip_code'].astype('category')

ured_zipcode_f2num_geo = ured_f2num.copy()
ured_zipcode_f2num_geo = pd.get_dummies(ured_zipcode_f2num_geo, columns=['city', 'state'], drop_first=True)


ured_zipcode_f2num_nogeo = ured_f2num.drop(columns=['city', 'state'])

X1 = ured_full_zipcode_geo.drop(columns=['price'])
y1 = ured_full_zipcode_geo['price']
X2 = ured_full_zipcode_nogeo.drop(columns=['price'])
y2 = ured_full_zipcode_nogeo['price']
X3 = ured_zipcode_f2cat_geo.drop(columns=['price'])
y3 = ured_zipcode_f2cat_geo['price']
X4 = ured_zipcode_f2cat_nogeo.drop(columns=['price'])
y4 = ured_zipcode_f2cat_nogeo['price']
X5 = ured_zipcode_f2num_geo.drop(columns=['price'])
y5 = ured_zipcode_f2num_geo['price']
X6 = ured_zipcode_f2num_nogeo.drop(columns=['price'])
y6 = ured_zipcode_f2num_nogeo['price']

scaler = StandardScaler()

X1 = scaler.fit_transform(X1)
X2 = scaler.fit_transform(X2)
X3 = scaler.fit_transform(X3)
X4 = scaler.fit_transform(X4)
X5 = scaler.fit_transform(X5)
X6 = scaler.fit_transform(X6)

nn_model('nn_model_200c_50s_zipcode_cat', X1, y1)
nn_model('nn_model_zipcode_cat', X2, y2)
nn_model('nn_model_200c_50s_zipcode_f2cat', X3, y3)
nn_model('nn_model_zipcode_f2cat', X4, y4)
nn_model('nn_model_200c_50s_zipcode_f2num', X5, y5)
nn_model('nn_model_zipcode_f2num', X6, y6)

from sklearn.tree import DecisionTreeRegressor
import joblib
def decision_tree_model(model_name, X, y, max_depth=None, min_samples_split=2, random_state=2137):
    os.makedirs("models", exist_ok=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Define and train the model
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, f"models/{model_name}.joblib")

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²:  {r2:.4f}")

decision_tree_model('dt_model_200c_50s_zipcode_cat', X1, y1)
decision_tree_model('dt_model_zipcode_cat', X2, y2)
decision_tree_model('dt_model_200c_50s_zipcode_f2cat', X3, y3)
decision_tree_model('dt_model_zipcode_f2cat', X4, y4)
decision_tree_model('dt_model_200c_50s_zipcode_f2num', X5, y5)
decision_tree_model('dt_model_zipcode_f2num', X6, y6)