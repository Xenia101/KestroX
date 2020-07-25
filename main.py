import load_data
import dtw_proto

if __name__ == "__main__":
    where = './user'
    kestrox_data = load_data.get_data()
    X_train, X_test, y_train, y_test = kestrox_data.ks_data(where)
