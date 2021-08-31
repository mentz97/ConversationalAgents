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

From inside the _simmc_ directroy,

```python
from dataset import SIMMCDataset
from preprocessing import preprocess
from model import SIMMCModel

train_dataset = SIMMCDataset(train=True, min_attribute_occ=5, concatenate=True, preprocess=preprocess)
validation_dataset = SIMMCDataset(train=False, concatenate=True, exclude_attributes=train_dataset.excluded_attributes, preprocess=preprocess)
test_dataset = SIMMCDataset(train=False, test=True, concatenate=True, exclude_attributes=train_dataset.excluded_attributes, preprocess=preprocess)

m = SIMMCModel()

savepath='./best_model' # to save and retrive the best model
EPOCHS=20
DEVICE='cuda'
LR=3e-5 # learning rate
WD=5e-5 # weight decay
SS=5 # step_size scheduler

m.train(train_dataset, validation_dataset, EPOCHS, DEVICE, LR, WD, SS, savepath=savepath)
m.validate(validation_dataset, DEVICE, savepath=savepath)
m.test(test_dataset, DEVICE, savepath=savepath)