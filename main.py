import opendatasets as od
import numpy as np
import pandas as pd
url = "https://www.kaggle.com/datasets/zeesolver/family"
od.download(url)
df = pd.read_csv('./family/Family Guy Dataset.csv')
print(df.head())