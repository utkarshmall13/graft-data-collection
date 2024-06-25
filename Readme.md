### Setup

See `setup.sh` for mamba/conda environment.

After setup run `mkdir data` to create a data directory where outputs will be stored.



### Download flickr metadata first
Create a flickr account and obtain an api key using this [link](https://www.flickr.com/services/api/misc.api_keys.html).
Once you have the key, use it in the following script.

> cd src
> python3 get_flickr_metadata.py

In the script it is only done for Hawaii, you'll have to use your own shapefiles for custom dataset. This will create a directory named `metadata`

### After metadata download use `subsample_dedup_bin.py` to get unique image metadata

> python3 subsample_dedup_bin.py

This will create another directory named `metadata_sdb`.

### After deduplicating download the flickr images using this metadata.

> python3 get_images_new.py

This will create another directory named `images`.

### Not all images will be downloaded, use `filter_existing_and_rename.py` to only filter down metadata of valid images.

> python3 filter_existing_and_rename.py

### Once we have the final metadata and images, use `satellite_sampler.py` to obtain center points of satellite images to be sampled.

> python3 satellite_sampler.py

This will give two outputs one is the image data `satellite_centers.pkl` contain center points to be sampled. 
The second thing it contains is satellite center to ground images mapping `image_data.pkl`. If you can get these two pickle piles you can go to the next step of downloading naip images.

### Use `naip_downloader.py` to download NAIP images from the location. 

If using earth engine you first need to create and earthengine project and authorize it. The best way to do it is using `authorization.ipynb`.

If using some other source of satellite you will have to modify this script for satellite images.

### Use `create_train_test_split.py` to create a non-intersecting train test split. 

This will create a many-to-one train-test splits useful for training GRAFT and other models.