import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,area_sqft,bedrooms,baths,property_type):
    try:
        loc_index = __data_columns.index(location.lower())
        home_index = __data_columns.index(property_type.lower())
    except:
        loc_index = -1
        home_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = baths
    x[1] = area_sqft
    x[2] = bedrooms
    if loc_index>=0 and home_index >= 0:
        x[loc_index] = 1
        x[home_index] = 1



    return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations
    global __home_types

    with open("artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[10:]  # first 3 columns are sqft, bath, bhk
        __home_types = __data_columns[3:10]


    global __model
    if __model is None:
        with open('artifacts/pakistan_house_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations
def get_home_types():
    return __home_types

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_home_types())
    print('Pkr',get_estimated_price('North Karachi'	,1089.00,2,	2, 'Flat'))

