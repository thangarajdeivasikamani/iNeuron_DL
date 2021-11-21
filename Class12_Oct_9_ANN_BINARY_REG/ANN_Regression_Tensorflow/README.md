# ANN- Regression implementation
ANN- Regression implementation - Predict the house price using below column

Use Early Stopping to Halt the Training of Neural Networks At the Right Time

1.  id	
1.  date	
1.  price	
1.  bedrooms	
1.  bathrooms	
1.  sqft_living	
1.  sqft_lot	
1.  floors	
1.  waterfront
1.  View	
1.  condition	
1.  grade	
1.  sqft_above	
1.  sqft_basement	
1.  yr_built	
1.  yr_renovated	
1.  zipcode	
1.  lat	
1.  long	
1.  sqft_living15
1.  sqft_lot15


## important commands ---------------

### creating envs -

```bash
conda create --prefix ./envs python=3.7 -y
```

### activate env

```bash
conda activate ./envs
```

## Reference -

* [Conda env commands](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)

### Setup command

```bash
pip install -e .
```
### Install the requirements
```bash
pip install requirements.txt
```
## Tensorboard log
```bash
tensorboard  --logdir logs_dir\tensorboard_logs\20211121_085029_tensorboard_log
```

### Build your own package for installation

```bash
python setup.py sdist bdist_wheel
```