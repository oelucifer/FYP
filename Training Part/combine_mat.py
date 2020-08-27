# this file combines the wiki_db.mat data to imdb_db.mat.
# now you can use a single imdb_db.mat file to train.

import scipy.io as sio
import numpy as np

imdb_data = sio.loadmat('data/imdb_db.mat')
wiki_data = sio.loadmat('data/wiki_db.mat')

gender_data = np.append(imdb_data['gender'],wiki_data['gender'])
gender_data = gender_data.reshape((1,-1))
imdb_data['gender'] = gender_data
del gender_data

age_data = np.append(imdb_data['age'],wiki_data['age'])
age_data = age_data.reshape((1,-1))
imdb_data['age'] = age_data
del age_data

db_data = np.append(imdb_data['db'],wiki_data['db'])
db_data = db_data.reshape((1,-1))
imdb_data['db'] = db_data
del db_data

image_data = np.append(imdb_data['image'],wiki_data['image'], axis = 0)
imdb_data['image'] = image_data
del image_data