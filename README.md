
## Datasets

We will use multiple datasets for this project but the following is a preliminary dataset from IEEE used to create 
the IEEE 1584-2018 standard for arc flash prediction and calulation. Good starting point

Need to create an account (free) with IEEE to access data

```
https://ieee-dataport.org/open-access/arc-flash-phenomena
```
Data is in excel files so needs to be processed before using

## Newer 2021 Data available in mv-data/

Before running any scripts, set up a Python virtual environment to avoid dependency conflicts.

#### Create virtual environment

```
python3 -m venv venv
```
#### Activate virtual environment


##### macOS / Linux

```
source venv/bin/activate
```

##### Windows (PowerShell)

```
venv\Scripts\Activate.ps1
```

### Install Requirements
```
pip install -r requirements.txt
```

## Training

To train, run:

```
python Models/resnet_regressor.py
```
To test and produce graphs, run:

```
python Models/load_and_predict.py
```




## TODOS:
1. ~~Find more datasets~~
2. ~~Find what factors are important in this data so we can integrate that into other models~~
3. ~~Select ML model options~~
4. ~~Implement load and predict and training using Rasp-Pi~~
5. optimize the resnet / other models for better inference and lower mem usage
6. Improve Dashboard Design
7. Test with the Schneider Hardware
8. 
