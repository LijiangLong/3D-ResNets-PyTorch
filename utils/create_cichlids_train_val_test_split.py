#make cichlids train and test list
import os
import numpy as np
import pandas as pd
import sys
import pdb


def main(split_ratio,test_datasets):
    train_list_file = '/Users/lijiang/Downloads/train.csv'
    val_list_file = '/Users/lijiang/Downloads/val.csv'
    test_list_file = '/Users/lijiang/Downloads/test.csv'
    dictionary_file = '/Users/lijiang/Downloads/category_dic'
    jpg_folder = '/Users/lijiang/Downloads/jpgs'
    project_labels_file = '/Users/lijiang/Downloads/ManualLabeledClusters.csv'
    project_labels_df = pd.read_csv(project_labels_file)
    project_labels_dic = {}
    for index, row in project_labels_df.iterrows():
        key = '_'.join([str(i) for i in [row['LID'],row['N'],row['t'],row['X'],row['Y']]])
        value = row['projectID']
        project_labels_dic[key] = value
        
    
#     pdb.set_trace()
    with open(train_list_file,'w') as train_output, open(val_list_file,'w') as val_output,open(test_list_file,'w') as test_output:
    #loop through all the videos and give it to either train or test based on the split ratio
        categories = {}
        i = 0
        for folder in os.listdir(jpg_folder):
            folder_path = jpg_folder +'/' + folder
            if not os.path.isdir(folder_path):
                continue
            i += 1
            categories[folder] = i
            for file in os.listdir(folder_path):
                if not os.path.isdir(folder_path+'/'+file):
                    continue
                output_string = folder + '/' + file + ' '
                if project_labels_dic[file.split('.')[0]] in test_datasets:
                    output_string += '\n'
                    test_output.write(output_string)
                    continue
                if np.random.uniform() < split_ratio:
                    output_string += str(i)+'\n'
                    train_output.write(output_string)
                
                else:
                    output_string += '\n'
                    val_output.write(output_string)
    with open(dictionary_file,'w') as output:
        for key,value in categories.items():
            output.write(str(value)+ ' '+ str(key)+'\n')



if __name__=='__main__':
    main(split_ratio = 0.9,test_datasets = ['MC6_5'])
