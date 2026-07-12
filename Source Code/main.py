import pandas as pd

train = pd.read_csv("data\MTS-Dialog-TrainingSet28SDHP%29.csv") #technically cuma perlu pake ini. validation gabakal dipake atau ngga di gabungin aja. krna this dataset was originally used for people fine tuning a model
validation = pd.read_csv("data\MTS-Dialog-Validation2029.csv")

print(train.head())

###Continued in IPYNB file for file exploration