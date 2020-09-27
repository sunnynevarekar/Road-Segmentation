# Road Segmentation from Satelite images using a custom Unet model

This repository contains code for training and evaluating a custom Unet model to segment roads from satelite images.

### Dataset
The dataset can be downloaded from this [kaggle link](https://www.kaggle.com/insaff/massachusetts-roads-dataset). 
Some of the images in training set does not have corresponding masks. The training code filters out those images.
All the images are of size 1500x1500. Code of data analysis is in [this notebook](notebooks/data_analysis.ipynb). 


### Model Architecture

<img src="resources/unet.png" width="400" height="800" align="center"/>


```main.py``` file contains code for training and testing.

E.g. 

For training 

```python 
python main.py train --train_dir=train_dir --model_save_path=path_to_save_model --batch_size=batch_size --epochs=num_epochs
```

To check all the parameters for training

```python 
python main.py train --help
```


For testing

```python 
python main.py test --test_dir=test_dir --checkpoint_path=trained_model_checkpoint_path --save_preds
```

To check all the parameters for testing

```python 
python main.py test --help
```




