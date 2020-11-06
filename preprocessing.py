import data_loader as dl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = dl.load_data('data/winequality-red.csv')  # import data file
df0 = dl.into_dataframe(data)  # change imported data into dataframe
df0_features = dl.get_features(df0, 'quality')
df0_labels = dl.get_labels(df0, 'quality')

# diving feature data into feature_train & feature_test

x = df0_features.to_numpy()

y = df0_labels.to_numpy().reshape(-1, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=27)

# normalizing feature_train & feature_test

# fit scaler on training data

norm_x = MinMaxScaler(feature_range=(0, 1)).fit(x_train)

# transform training data

x_train = norm_x.transform(x_train)


# transform testing dataa

x_test = norm_x.transform(x_test)

norm_y = MinMaxScaler(feature_range=(0, 1)).fit(y_train)

y_train = norm_y.transform(y_train)

y_test = norm_y.transform(y_test)
