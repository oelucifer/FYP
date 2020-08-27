use the following commands in the order.

Note: Please Use Colab...

The download.sh script downloads the imdb and wiki dataset and stores them in
data folder.

./download.sh

-------------------------------------

These commands create the mat files that will be given to training script.
the larger the img_size the more the accuracy and more the space it will take
in the RAM.

python3 create_db.py --output data/imdb_db.mat --db imdb --img_size 64
python3 create_db.py --output data/wiki_db.mat --db wiki --img_size 64

-------------------------------------

the combine_mat script combines the both imdb_db.mat and wiki_db.mat files 
to a single imdb_db.mat file
python3 combine_mat.py

-------------------------------------

the training script takes the mat file as an input....
python3 train.py --input data/imdb_db.mat