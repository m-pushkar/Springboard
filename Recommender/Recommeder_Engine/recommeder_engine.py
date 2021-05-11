import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import os.path
from sklearn.metrics.pairwise import linear_kernel

# for distance calculations
from geopy.distance import great_circle
from geopy.geocoders import Nominatim


class Recommender_Engine:
    ###----------------------------------------Keyword Search Recommender--------------------------------------------###
    def __init__(self, n=5, stars_original=False):
        ###-------------------------------------------Import datasets------------------------------------------------###
        self.business = pd.read_csv('clean_business.csv')
        self.business['postal_code'] = self.business.postal_code.astype(str)
        self.review = pd.read_csv('clean_review.csv')
        self.review_s = self.review[self.review.business_id.isin(self.business.business_id.unique())]

        mean_global = ((self.business.stars * self.business.review_count).sum() / (self.business.review_count.sum()))
        k = 30
        self.business['stars_adj'] = ((self.business.review_count * self.business.stars) \
                                      + (k * mean_global)) / (self.business.review_count + k)

        """
        Instantiate the object. Default setting for ranking would be stars_adj,
        to rank by original stars set stars_original=True.
        """
        self.n = n  # Number of recommendations
        self.stars_original = stars_original  # Boolean for ranking method
        self.disply_columns = ['name', 'address', 'city', 'state', \
                               'attributes.RestaurantsPriceRange2', \
                               'review_count', 'stars', 'stars_adj', \
                               'cuisine', 'style']  # List of columns to be displayed in the results

        if self.stars_original:
            score = 'stars'
        else:
            score = 'stars_adj'

        # Filter only open restaurants
        self.recommendation = business[business.is_open == 1].sort_values(score, ascending=False)

        # Pre-load pickle files
        if personalized:

            # For collaborative module
            with open('svd_algo_trained_info', 'rb') as f:
                useful_info = pickle.load(f)

            # For content based module



    def filter_location(self):
        """
        Filter recommendations by user's location. Matching restaurant is the restaurant within the acceptable distance of the location of interest.
        """
        from geopy.geocoders import Nominatim
        from geopy.distance import great_circle

        geolocator = Nominatim(user_agent="Recommendation")
        address = [self.city, self.state, self.zipcode]
        address = ",".join([str(i) for i in address if i != None])
        location = geolocator.geocode(address, timeout=10)

        # Calculate recommendations distance and append a column
        self.recommendation['distance_recommendations'] = self.recommendation.apply \
            (lambda row: (great_circle((row.latitude, row.longitude), (location.latitude, location.longitude)).miles),
             axis=1)

        self.disply_columns.insert(0, 'distance_recommendations')
        self.recommendation = self.recommendation[self.recommendation.distance_recommendations <= self.distance_max]

    def filter_state(self):
        self.recommendation = self.recommendation[self.recommendation.state == self.state]

    def filter_price(self):
        self.recommendation = self.recommendation[self.recommendation \
            ['attributes.RestaurantsPriceRange2'].isin(self.price)]

    def filter_cuisine(self):

        idx = []
        for i in self.recommendation.index:
            if self.recommendation.loc[i, 'cuisine'] is not np.nan:
                entry = self.recommendation.loc[i, 'cuisine']
                entry = str(entry).split(',')
                if self.cuisine in entry:
                    idx.append(i)
        self.recommendation = self.recommendation.loc[idx]

    def filter_style(self):

        idx = []
        for i in self.recommendation.index:
            if self.recommendation.loc[i, 'style'] is not np.nan:
                entry = self.recommendation.loc[i, 'style']
                entry = str(entry).split(',')
                if self.style in entry:
                    idx.append(i)
        self.recommendation = self.recommendation.loc[idx]

    def display(self):

        if len(self.recommendation) == 0:
            print("Sorry, there are no matching recommendations.")
        elif self.n < len(self.recommendation):
            print("Below is the list of the top {} recommended restaurants for you: ".format(self.n))
            print(self.recommendation.iloc[:self.n][self.disply_columns])
        else:
            print("Below is the list of the top {} recommended restaurants for you: ".format(len(self.recommendation)))
            print(self.recommendation.iloc[self.disply_columns])

    def keyword_filtering(self, catalog=business[business.is_open == 1], price=None, \
                          zipcode=None, city=None, state=None, distance_max=10, cuisine=None, style=None):

        self.recommendation = catalog  # Set restaurant catalog
        self.recommendation['distance__recommendations'] = np.nan  # Reset distance
        self.disply_columns = ['name', 'address', 'city', 'state', \
                               'attributes.RestaurantsPriceRange2', \
                               'review_count', 'stars', 'stars_adj', \
                               'cuisine', 'style']  # Reset columns

        # Based on keyword search
        self.zipcode = zipcode
        self.city = city
        self.state = state
        self.distance_max = distance_max
        self.cuisine = cuisine
        self.style = style
        self.price = price

        # Filter_location
        if (self.zipcode != None) or (self.city != None) or (self.state != None):
            if (self.zipcode != None) or (self.city != None):
                self.filter_location()
            elif (self.state != None):
                self.filter_state()
            if len(self.recommendation) == 0:
                print("Sorry, there are no matching recommendations.")

        # Filter_price
        if self.price != None:
            self.price = [i.strip() for i in price.split(',')]  # Multiple inputs
            self.filter_price()
            if len(self.recommendation) == 0:
                print("Sorry, there are no matching recommendations.")

        # Filter_cuisine
        if self.cuisine != None:
            self.filter_cuisine()
            if len(self.recommendation) == 0:
                print("Sorry, there are no matching recommendations.")

        # Filter_style
        if self.style != None:
            self.filter_style()
            if len(self.recommendation) == 0:
                print("Sorry, there are no matching recommendations.")

        # Sort recommendations by user input for ranking method
        if self.stars_original:
            score = 'stars'
        else:
            score = 'stars_adj'

        self.recommendation = self.recommendation.sort_values(score, ascending=False)

        # Display recommendations
        self.display()

        return self.recommendation

    ###------------------------------------------Collaborative Recommeder--------------------------------------------###
    def collaborative_filtering(self, user_id=None):
        self.user_id = user_id
        if self.user_id is None:
            print('User ID is not provided')
            return None
        if len(user_id) != 22:  # Sanity check on length of user id
            print('Invalid user ID')
            return None

        self.recommendation = business[business.is_open == 1]
        if 'stars_pred' in self.recommendation.columns:
            self.recommendation.drop('stars_pred', axis=1, inplace=True)

        self.display_columns = ['name', 'address', 'city', 'state', \
                                'attributes.RestaurantsPriceRange2', \
                                'review_count', 'stars', 'stars_adj', \
                                'cuisine', 'style']

        # Useful info from trained model
        mean_rating = useful_info['mean_rating']
        user_latent = useful_info['user_latent']
        item_latent = useful_info['item_latent']
        user_bias = useful_info['user_bias']
        item_bias = useful_info['item_bias']
        userid_idx = useful_info['userid_to_index']
        itemid_idx = useful_info['itemid_to_index']

        # Recommendations
        if self.user_id in userid_idx:
            u_idx = userid_idx[self.user_id]
            pred = mean_rating + user_bias[u_idx] + item_bias + np.dot(user_latent[u_idx, :], item_latent.T)
        else:
            print('Sorry, no personlaized recommendations yet!')
            print('\nHere are generic recommendations: ')

            pred = mean_rating + item_bias

        prediction = pd.DataFrame(data=pred, index=itemid_idx.values(), columns=['stars_pred'])
        prediction.index.name == 'matrix_item'
        assert len(prediction) == len(pred)
        prediction['business_id'] = list(itemid_idx.keys())

        # Filter to unrated business by user
        if self.user_id in userid_idx:
            rated_bus = review[review.user_id == self.user_id].business_id.unique()
            prediction = prediction[~prediction.business_id.isin(rated_bus)]

        self.recommendation = self.recommendation.merge(prediction, on='business_id', how='inner')
        self.recommendation = self.recommendation.sort_values('stars_pred', ascending=False).reset_index(drop=True)
        self.display_columns.insert(0, 'stars_pred')
        self.display()

        return self.recommendation

