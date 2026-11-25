import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('mieszkania_okolice_warszawy.csv')
np.random.seed(21)
df.loc[np.random.rand(len(df)) < 0.05, 'pokoje'] = np.nan
df.loc[np.random.rand(len(df)) < 0.05, 'odleglosc_pkp_km'] = np.nan

X = df.drop(columns=['cena_pln'])
y = df['cena_pln']

kolumny_liczbowe = ['metraz', 'pokoje', 'wiek_budynku', 'odleglosc_pkp_km']
kolumny_tekstowe = ['miasto']

#Preprocessing data
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, kolumny_liczbowe), ('cat', categorical_transformer, kolumny_tekstowe)])

#Creating model
model = RandomForestRegressor(n_estimators=100, random_state=0)

#Creating Pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

my_pipeline.fit(X, y)

joblib.dump(my_pipeline, 'pipeline_mieszkania.pkl')

scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Average MAE score for the pipeline is: {scores.mean():,.0f} zł")
