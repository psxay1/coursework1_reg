import data_loader as dl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = dl.load_data('data/winequality-red.csv')  # import data file
df0 = dl.into_dataframe(data)  # change imported data into dataframe
df0_features = dl.get_features(df0, 'quality')

df0_labels = dl.get_labels(df0, 'quality')

# diving feature data into feature_train & feature_test

x = df0_features

y = df0_labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)

# normalizing feature_train & feature_test

# fit scaler on training data

norm = MinMaxScaler().fit(x_train)

# transform training data

x_train_norm = norm.transform(x_train)

# transform testing dataa

x_test_norm = norm.transform(x_test)

