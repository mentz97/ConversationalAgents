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

To run tests and scripts you must source the virtual environment every time you open a new console session.

```bash
(.env) $ pytest

============================= test session starts ==============================
platform darwin -- Python 3.9.5, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: /Users/pettinz/Developer/machine_learning/ML_project
collected 3 items                                                                                                                                                                                                                                                                                                

simmc/dataset_test.py ...                                                 [100%]

============================== 3 passed in 7.20s ===============================
```