import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

#Category_encoder is needed for target encoding text

try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder=None
    print("Warning: category_encoders not installed Target will ")

def main():

    print("Loading Datasets")
    file_path='Train.csv'

    if not os.path.exists(file_path):
        print(f"Error: Cannot find {file_path}")
        return
    df=pd.read_csv(file_path)
    print(f"Dataset loded successfully Rows:{df.shape[0]},Features:{df.shape[1]}")


    #Handling the missing data
    print("Handling the  Missing data")
    print("Artificially deleting some 'Hits' (H) data to demonstrate ")
    
    df.loc[0:25,'H']=np.nan
    imputer=SimpleImputer(strategy='median')

    df['H']=pd.to_numeric(df['H'], errors='coerce')
    df['H']=imputer.fit_transform(df[['H']]).ravel()
    print(f"imputation complete. 'Hits' (H) now has {df['H'].isnull().sum()} missing values")
    
    
if __name__ =="__main__":
    main()


    
