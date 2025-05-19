import dmp_functions as dmp
import pandas as pd
from scipy import stats
import numpy as np

ured = pd.read_csv('data/usa-real-estate-dataset.csv')
uscities = pd.read_csv('data/uscities.csv')

dmp.print_dataset_overview(ured, 'USA Real Estate Dataset')

ured_filtered = ured.drop(columns=['brokered_by', 'street', 'status'])
dmp.print_dataset_overview(ured_filtered, 'USA Real Estate Dataset - filtered')
target = ured_filtered['price']
dmp.plot_boxplot(ured_filtered, 'bed')
dmp.plot_boxplot(ured_filtered, 'bath')
dmp.plot_boxplot(ured_filtered, 'acre_lot')
dmp.plot_boxplot(ured_filtered, 'house_size')
dmp.plot_scatterplot(ured_filtered, 'bed', 'bath')
ured_filtered = ured_filtered[
    ((ured_filtered['bath'] < 2 * ured_filtered['bed']) |
     (ured_filtered['bed'].isna()) |
      (ured_filtered['bath'].isna())) & (ured_filtered['bed'] < 100)
        ]
dmp.plot_scatterplot(ured_filtered, 'bed', 'bath')
ured_filtered = ured_filtered[ured_filtered['acre_lot'] != 0]
ured_filtered = ured_filtered[ured_filtered['house_size'] != 0]
dmp.plot_scatterplot(ured_filtered, 'acre_lot', 'house_size', y_log=True)
ured_outliers_fixed = dmp.remove_outliers_3s(ured_filtered, 'house_size')
ured_filtered = ured_outliers_fixed
dmp.plot_scatterplot(ured_filtered, 'acre_lot', 'house_size')
ured_filter_date = ured_filtered
ured_filter_date = ured_filter_date[
    (ured_filter_date['prev_sold_date'].isnull()) |
    (ured_filter_date['prev_sold_date']
     .str.slice(0, 4)
     .apply(pd.to_numeric, errors='coerce')
     .between(1900, 2025))
]
ured_filtered = ured_filter_date
dmp.count_nans(ured_filtered)
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
dmp.count_nans(ured_filtered)

dmp.draw_histogram(ured_filtered['bed'], title='Bed frequency', xlabel='Bed', ylabel='Frequency')
dmp.draw_histogram(ured_filtered['bath'], title='bath frequency', xlabel='bath', ylabel='Frequency')
dmp.draw_histogram(ured_filtered['acre_lot'], title='acre_lot frequency', xlabel='acre_lot', ylabel='Frequency')
dmp.draw_histogram(ured_filtered['house_size'], title='house_size frequency', xlabel='house_size', ylabel='Frequency')
dmp.draw_histogram(ured_filtered['years_since_sold'], title='years_since_sold frequency', xlabel='years_since_sold', ylabel='Frequency')
ured_scaled = ured_filtered.copy()
ured_scaled['house_size'] = (ured_filtered['house_size'] - ured_filtered['house_size'].min()) / (ured_filtered['house_size'].max() - ured_filtered['house_size'].min())
ured_scaled['acre_lot'] = (ured_filtered['acre_lot'] - ured_filtered['acre_lot'].min()) / (ured_filtered['acre_lot'].max() - ured_filtered['acre_lot'].min())
dmp.draw_histogram(ured_scaled['house_size'], title='house_size frequency', xlabel='house_size', ylabel='Frequency')
dmp.draw_histogram(ured_scaled['acre_lot'], title='acre_lot frequency', xlabel='acre_lot', ylabel='Frequency')


ured_normalized = ured_scaled.copy()

ured_normalized['house_size'], lambda_house_size = stats.boxcox(ured_scaled['house_size'] + 1)

dmp.draw_histogram(ured_normalized['house_size'], title='house_size frequency', xlabel='house_size', ylabel='Frequency')
dmp.print_dataset_overview(uscities, 'USA cities')
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
dmp.print_dataset_overview(ured_normalized, 'USA cities')
ured_normalized['city_population'] = (ured_normalized['city_population'] - ured_normalized['city_population'].min()) / (ured_normalized['city_population'].max() - ured_normalized['city_population'].min())
ured_normalized.to_csv('usa-real-estate-dataset-normalized.csv', index=False)

from sklearn.preprocessing import OneHotEncoder
categorical_cols = ['state', 'city', 'zip_code']
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
encoded_features = enc.fit_transform(ured_normalized[categorical_cols])
encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_features)
encoded_df = encoded_df.add_prefix('encoded_')
ured_encoded = pd.concat([ured_normalized, encoded_df], axis=1)
ured_encoded = ured_encoded.drop(categorical_cols, axis=1)

