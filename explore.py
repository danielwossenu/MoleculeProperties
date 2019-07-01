import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
train_data = pd.read_csv("data/train.csv")

# transform_data(input_data)
magshield_data = pd.read_csv("data/magnetic_shielding_tensors.csv")
dipole_data = pd.read_csv("data/dipole_moments.csv")
pe_data = pd.read_csv("data/potential_energy.csv")
mulliken_data = pd.read_csv("data/mulliken_charges.csv")
scalar_data = pd.read_csv("data/scalar_coupling_contributions.csv")


other_dfs = [magshield_data, dipole_data, pe_data, mulliken_data, scalar_data]

for df in other_dfs:
    train_columns = train_data.columns
    df_columns = df.columns

    for df_col in df_columns:
        if df_col not in train_columns:
            train_data[df_col] = df[df_col]

# train_data.join(magshield_data, on='molecule_name', how='left')

print(train_data.head())

print(train_data.dtypes)

#____create a function that creates this pipeline and saves the the label encoders ____

#separate out the columsn that need to be one hot encoded into a separate df
catagorical_col_names = ["atom_index_0","atom_index_1", "type"]

cat_df = train_data[catagorical_col_names]


for cat_name in catagorical_col_names:
#     cat_df[cat_name] = train_data[cat_name]
    train_data = train_data.drop([cat_name], axis=1)
    # train_data.drop(columns=[cat_name])

train_data = train_data.drop(['id','molecule_name'], axis=1)

#train encoders and do the tranformations
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

cat_array = enc.fit_transform(cat_df.values)

#convert to numpy array and join with the df that has the numerical train data
y_df = train_data[['scalar_coupling_constant']]
train_data = train_data.drop(['scalar_coupling_constant'], axis=1)


for i,feature in enumerate(enc.get_feature_names().tolist()):
    print(feature)
    train_data[feature] = cat_array[:,i]





# full_train_data = np.concatenate([train_data.values, cat_array], axis=1)

# train_data.drop


X_train, X_test, y_train, y_test = train_test_split(train_data.values[:50000,:], y_df.values[:50000,:], test_size=0.33, random_state=36)

print(X_train.shape)
#set up model/s

model = MLPRegressor(hidden_layer_sizes=(100,100),verbose=True)

#train with 10 fold cross validation or possible separate out a validation set

model.fit(X_train, y_train)

preds = model.predict(X_test)


mae = mean_absolute_error(y_test, preds)

log_mae = np.log(mae)

mean_log_mae = np.mean(log_mae)

print(mean_log_mae)
print('a')