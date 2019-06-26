############# Configuration file #############

# Name of dataset
# name = 'INBREAST'
# name = 'INBREAST_b'
# name = 'GURO_ALL'
# name = 'GURO_TRAIN'
# name = 'GURO_TEST'
# name = 'GURO_CAD'
# name = 'GURO_CAD_b'
# name = 'image_patches_guro'
# name = 'image_patches_inbreast_new'
# name = 'image_patches_guro_new'
# name = 'inbreast_patches_9'
name = 'test'
# name = 'GURO_MORE_BENIGN'
# name = 'GURO_FILTER'
# name = 'GURO_ALL'

# Base directory for data formats
data_base = '/home/qtt/datasets/' + name

# Base directory for augmented data formats
resize_base = '/home/qtt/datasets/' + name + '/resized'
split_base = '/home/qtt/datasets/' + name + '/split'

# Directory for data formats
resize_dir = resize_base 
split_dir = split_base

# Validation split
val_num = 0
