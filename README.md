# Road Segmentation from Satelite images using a custom Unet model

This repository contains code for training and evaluating a custom Unet model to segment roads from satelite images.


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




