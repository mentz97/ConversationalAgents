# ML_project
## Requirements
- Python 3.9.5
- PiP
- venv
## How to use

Clone the **repository**

```bash
$ git clone https://github.com/mentz97/ML_project.git
$ cd ML_project
$ git checkout nopack
```

Create and setup the **virtual environment**. To check whether the steps below have been successful or not, the console notify you if you are currently on a virtual environment or not.

```bash
$ python3 -m venv .env
$ source .env/bin/activate
(.env) $ python3 -m pip install -r requirements.txt
```

To close the virtual environment you just need to exit the console session
```bash
(.env) $ exit
```

## How to run

Inside the _simmc_ directroy,

```python
from dataset import SIMMCDataset
from model import SIMMCModel



train_dataset = SIMMCDataset(train=True, min_attribute_occ=2, concatenate=True)
validation_dataset = SIMMCDataset(train=False, exclude_attributes=train_dataset.excluded_attributes)
# test_dataset = ...

m = SIMMCModel()
m.train(train_dataset, validation_dataset)
# m.validate(test_dataset)
```

## To be tested

- [x] API loader
- [x] Dataset creation
- [ ] `.tolist()` _lines 152,153,166,167_ of _model.py_
- [ ] `model_actions` generation _lines 174-190_ of _model.py_
- [ ] no `val_dataloader` in _test_ function in _model.py_. Why using a test function if there is the validate?