import data_loader as dl
import preprocessing as pp
global df0, data
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = dl.load_data('data/winequality-red.csv')          # import data file

    df0 = dl.into_dataframe(data)      # change imported data into dataframe
    # print(df0)

    df0_features = dl.get_features(df0, 'quality')
    # print(df0_features)

    df0_labels = dl.get_labels(df0, 'quality')
    # print(df0_labels)

    norm_feature_train = pp.x_train_norm
    norm_feature_test = pp.x_train_norm

