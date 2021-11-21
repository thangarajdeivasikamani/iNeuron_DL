# ANN BINARY CLASSIFICATION
Predict the customer churn based on given column

- RowNumber	
- CustomerId	
- Surname	
- CreditScore	
- Geography	
- Gender	
- Age	
- Tenure	
- Balance	
- NumOfProducts
- HasCrCard	
- IsActiveMember	
- EstimatedSalary	
- Exited



## important commands ---------------


### creating envs -

```bash
conda create --prefix ./envs python=3.7 -y
```

### activate env

```bash
conda activate ./envs
```
### Install the requirements
```bash
pip install -e .
pip install requirements.txt
```
## Tensorboard log
```bash
tensorboard  --logdir logs_dir\tensorboard_logs\20211121_085029_tensorboard_log
```
## Reference -

* [Conda env commands](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)