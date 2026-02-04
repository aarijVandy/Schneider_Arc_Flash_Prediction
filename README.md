
## Datasets

We will use multiple datasets for this project but the following is a preliminary dataset from IEEE used to create 
the IEEE 1584-2018 standard for arc flash prediction and calulation. Good starting point

Need to create an account (free) with IEEE to access data

```
https://ieee-dataport.org/open-access/arc-flash-phenomena
```
Data is in excel files so needs to be processed before using

## Newer 2021 Data available in mv-data/

To train, run: (needs you environment to be setup. use conda to create with requirements.txt to include packages)

```
python Models/resnet_regressor.py
```
To test and produce graphs, run:

```
python Models/load_and_predict.py
```




## TODOS:
1. ~~Find more datasets~~
2. Find what factors are important in this data so we can integrate that into other models
3. ~~Select ML model options~~
4. Implement load and predict and training using Rasp-Pi
5. optimize the resnet / other models for better inference and lower mem usage
6. 
