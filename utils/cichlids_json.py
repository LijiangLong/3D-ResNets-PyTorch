from __future__ import print_function, division
import os
import sys
import json
import pandas as pd
import pdb

def convert_csv_to_dict(csv_path, subset):
    try:
        data = pd.read_csv(csv_path, delimiter=' ', header=None)
    except:
        print('Warning: no {}, check data'.format(csv_path))
        return {}
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.ix[i, :]
        slash_rows = data.ix[i, 0].split('/')
        class_name = slash_rows[0]
        basename = slash_rows[1]
        
        keys.append(basename)
        key_labels.append(class_name)
        
    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        database[key]['annotations'] = {'label': label}
    
    return database

def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.ix[i, 1])
    return labels

def convert_cichlids_csv_to_activitynet_json(label_csv_path, train_csv_path, 
                                           val_csv_path,test_csv_path, dst_json_path):
    labels = load_labels(label_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training')
    val_database = convert_csv_to_dict(val_csv_path, 'validation')
    test_database = convert_csv_to_dict(test_csv_path, 'test')
    
    dst_data = {}
    
    dst_data['labels'] = labels
    
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    dst_data['database'].update(test_database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

if __name__ == '__main__':
    csv_dir_path = sys.argv[1]
    label_csv_path = os.path.join(csv_dir_path, 'category_dic')
    train_csv_path = os.path.join(csv_dir_path, 'train.csv')
    val_csv_path = os.path.join(csv_dir_path, 'val.csv')
    test_csv_path = os.path.join(csv_dir_path, 'test.csv')
    dst_json_path = os.path.join(csv_dir_path, 'cichlids.json')

    convert_cichlids_csv_to_activitynet_json(label_csv_path, train_csv_path,
                                               val_csv_path, test_csv_path,dst_json_path)
