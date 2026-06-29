
import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os


try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder=None
    print("Warning:category_encoders not installed.Target Encoding will ")

def main():

    print("Loading Datasets:")
    # Look for train.csv in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Train.csv')

    if not os.path.exists(file_path):
        print(f" Error cannot find '{file_path}'")
        return 
    df=pd.read_csv(file_path)
    print(f"Dataset Loaded Succesfully.Rows:{df.shape[0]},Features:{df.shape[1]}\n")
    print("HANDLING MISSING DATA")
    print("Artificially Deleting some 'Hits' (H) data to demonstrate creating missing data for the lesson.")

    df.loc[0:25,'H']=np.nan
    imputer =SimpleImputer(strategy="median")
    df['H'] = imputer.fit_transform(df[['H']].apply(pd.to_numeric, errors='coerce'))
    print(f"Imputation complete .'Hits (H) now has {df['H'].isnull().sum()} null values.\n")

    print("Evaluating the skewness of the Runs (R) disturbution...")
    df["LogRuns"] = np.log1p(pd.to_numeric(df["R"], errors='coerce'))
    print(f"Log Transformation applied. New skewness:{df['LogRuns'].skew():.2f}(closer to 0 is perfectly balanced).\n") 
    df["Team_ID"]=["Team_"+str(np.random.randint(1,150)) for _ in range(len(df))]
    if TargetEncoder is not None:
        print("Applying Target Encoder")
        encoder = TargetEncoder()
        df["Team_ID_Encoded"]=encoder.fit_transform(df["Team_ID"],df['W'])

    # Feature selection and model training
    features_to_rent = ["R", "HR", "SO", "SB"]
    # Convert features to numeric, coerce errors
    X_features = df[features_to_rent].apply(pd.to_numeric, errors='coerce')
    # Convert target to numeric
    y_target = pd.to_numeric(df["W"], errors='coerce')
    # Drop rows with any NaN values
    mask = X_features.notnull().all(axis=1) & y_target.notnull()
    X_valid = X_features[mask]
    y_valid = y_target[mask]

    selector = SelectKBest(score_func=mutual_info_regression, k=2)
    selector.fit(X_valid, y_valid)
    winning_features = selector.get_support()
    best_features = X_features.columns[winning_features].tolist()
    print("Selected features:", best_features)

    # Splitting Data (using cleaned numeric data)
    X = X_valid[best_features]
    y = y_valid

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training Data size :{X_train.shape}")
    print(f"Testing Data size:{X_test.shape}\n")

    #Training Model 
    model=LinearRegression()
    model.fit(X_train,y_train)

    predictions=model.predict(X_test)
    print(predictions)


    # Comparig model prediction to the actual real answer

    actual_wins=y_test.head(5).values
    predicted_wins=predictions[:5]

    for i in range(5):
        predicted=round(predicted_wins[i])
        actual=actual_wins[i]
        differences=abs(actual-predicted)

        print(f"Model Guessed :{predicted}")
        print(f"Real answer :{actual}")
        print(f"Differences :{differences}")

if __name__ == '__main__':
    main()    