
### nyctaxi.py
### Kelly Spendlove
### 2020-07-07
### MIT LICENSE

"""
module for loading and cleaning NYC taxi
"""

import numpy as np
import pandas as pd 

def load_data (nrows=1_000_000):
	path = '/Users/kelly/Downloads/new-york-city-taxi-fare-prediction'
	try:
		return pd.read_csv(path+'/train.csv', nrows=nrows,  parse_dates = ['pickup_datetime'])
	except:
		print('Error importing data.')

def remove_missing ( df ):
	return df.dropna(how='any', axis='rows')

def remove_by_fare ( df ):
	return df[df.fare_amount >= 2.5]

def in_bounding_box ( df, X, Y):
    return (df . pickup_longitude >= X[0]) & (df . pickup_longitude <= X[1]) \
         & (df . pickup_latitude >= Y[0]) & (df . pickup_latitude <= Y[1]) \
         & (df . dropoff_longitude >= X[0]) & (df . dropoff_longitude <= X[1]) \
         & (df . dropoff_latitude >= Y[0]) & (df . dropoff_latitude <= Y[1])

def remove_by_gps (df,  X= [-74.3, -72.9], Y = [40.5, 41.8] ):
	return df[in_bounding_box (df, X, Y)]

def remove_by_passenger_count ( df, max_count=6 ):
    return df[(df.passenger_count > 0) & (df.passenger_count <= max_count)]

def clean_data ( df ):
	df = remove_missing(df)
	df = remove_by_fare(df)
	df = remove_by_gps(df)
	df = remove_by_passenger_count(df)
	return df

# calculate distance in mi based on gps coordinates
def geodesic (phi_1, lam_1, phi_2, lam_2, r = 3958.8 ):
    phi_1, lam_1, phi_2, lam_2 = map(np.radians, [phi_1, lam_1, phi_2, lam_2])
    d_lats, d_longs = phi_2-phi_1, lam_2-lam_1
    a = np.sin(d_lats/2.0)**2 + np.cos(phi_1)*np.cos(phi_2)*np.sin(d_longs/2.0)**2
    return 2.0 * r * np.arcsin ( np.sqrt ( a ) )

def add_distance ( df ):
	 df['distance'] = geodesic (df['pickup_latitude'], df['pickup_longitude'], 
	 							df['dropoff_latitude'], df['dropoff_longitude'] )


def remove_by_distance ( df, min_d = .5, max_d = 21 ):
	return df[(df.distance > min_d) | (df.fare_amount <= max_d)]

def add_and_clean_distance ( df ):
	add_distance ( df )
	return remove_by_distance ( df )


