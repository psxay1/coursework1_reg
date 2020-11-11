import data_loader as dl
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# data = dl.load_data('data/winequality-red.csv')  # import data file
data = dl.load_data('data/winequality-white.csv')  # import data file
df0 = dl.into_dataframe(data)  # change imported data into dataframe
df0_features = dl.get_features(df0, 'quality')
df0_labels = dl.get_labels(df0, 'quality')

#-----------------------------------------------------------------------------
# data for kFold
features = df0_features.to_numpy()
kFold_features = MinMaxScaler().fit_transform(features)
labels = df0_labels.to_numpy().reshape(-1, 1)
kFold_labels = MinMaxScaler().fit_transform(labels)
print("kbsojbsosr")
#-----------------------------------------------------------------------------
# data for testing
features_test = kFold_features[0:500]
labels_test = kFold_labels[0:500]

plot_features = df0_features.to_numpy()[0:500]