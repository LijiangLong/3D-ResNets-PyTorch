#make cichlids train and test list
import os
import numpy as np
import pandas as pd
import sys
import pdb


def main(split_ratio,test_datasets):
    home_folder = '/data/home/llong35/data/11_07_2019'
    train_list_file = os.path.join(home_folder,'train.csv')
    val_list_file = os.path.join(home_folder,'val.csv')
    test_list_file = os.path.join(home_folder,'test.csv')
    dictionary_file = os.path.join(home_folder,'category_dic')
    jpg_folder =  os.path.join(home_folder,'annotate_video_jpg')
    project_labels_file =  os.path.join(home_folder,'ManualLabeledClusters.csv')
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
#     main(split_ratio = 0.9,test_datasets = ['MC6_5','CV10_3','MC16_2','MCxCVF1_12a_1','MCxCVF1_12b_1','TI2_4','TI3_3'])
    main(split_ratio = 0.9,test_datasets = ['MC6_5'])
