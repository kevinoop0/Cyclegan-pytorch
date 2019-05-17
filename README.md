# The pytorch implemention of Cycle-gan

This is a Pytorch implementation of the "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"  [paper](<https://arxiv.org/abs/1703.10593>).

 

## Dataset

you can download data from [link](<https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>) , or run the script `get_data.py` to choose dataset you want download.

```python
python get_data.py
```

- `facades`: 400 images from the [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade). [[Citation](datasets/bibtex/facades.tex)]
- `cityscapes`: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com). [[Citation](datasets/bibtex/cityscapes.tex)]
- `maps`: 1096 training images scraped from Google Maps.
- `horse2zebra`: 939 horse images and 1177 zebra images downloaded from [ImageNet](http://www.image-net.org) using keywords `wild horse` and `zebra`
- `apple2orange`: 996 apple images and 1020 orange images downloaded from [ImageNet](http://www.image-net.org) using keywords `apple` and `navel orange`.
- `summer2winter_yosemite`: 1273 summer Yosemite images and 854 winter Yosemite images were downloaded using Flickr API. See more details in our paper.
- `monet2photo`, `vangogh2photo`, `ukiyoe2photo`, `cezanne2photo`: The art images were downloaded from [Wikiart](https://www.wikiart.org/). The real photos are downloaded from Flickr using the combination of the tags *landscape* and *landscapephotography*. The training set size of each class is Monet:1074, Cezanne:584, Van Gogh:401, Ukiyo-e:1433, Photographs:6853.
- `iphone2dslr_flower`: both classes of images were downlaoded from Flickr. The training set size of each class is iPhone:1813, DSLR:3316. See more details in our paper.

To train a model on your own datasets, you need to create a data folder as the following. (orange2apple is dataset name)

```
data/
	orange2apple/
			train/
				 A/...jpg
				 B/...jpg
			 test/
				 A/...jpg
				 B/...jpg
```



## Usage
`python cyclegan.py ARGS`

Possible ARGS are:
+  `--epochs` number of epochs of training, default is 200;
+  `--dataset_name` name of the dataset, default is "orange2apple";
+  `--batch_size` size of the batches, default is 1;
+  `--lr` adam: learning rate, default is 0.0002;
+  `--decay_epoch` epoch from which to start lr decay, default is 100;
+  `--img_height` size of image height (default is `128`);
+  `--img_width` size of image width (default is `128`);
+  `--sample_interval` interval between saving generator outputs(default is `100`) ;
+  `--checkpoint_interval` interval between saving model checkpoints (default is `-1`) ;
+  `--n_residual_blocks` number of residual blocks in generator (default is `9`) ;

An example:

```python
python cyclegan.py --n_residual_blocks 6 
```



## Result



<img src="http://m.qpic.cn/psb?/V12kySKV4IhBFe/Jr7RB08aPdzYOXwf7Ye*G8cMrJSgX4*BdB3TRtviLLw!/b/dMAAAAAAAAAA&bo=owLOAAAAAAARB18!&rf=viewer_4">
