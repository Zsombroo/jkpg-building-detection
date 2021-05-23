"""
Creates random folds of data and copy files to it's respective folder.
"""

import os
import shutil
import re

import cross_validation_set_generator

def copy_files(PATH_TO_DATA):
    pattern = re.compile('.*\.png$')
    list_of_files = [file.split('.')[0] for file in os.listdir(PATH_TO_DATA) if pattern.match(file)]

    folds = cross_validation_set_generator.get_cross_validation_sets(list_of_files, 5)
    
    for idx, fold in enumerate(folds):
        if not os.path.exists(PATH_TO_DATA+'/fold_'+str(idx+1)):
            os.makedirs(PATH_TO_DATA+'/fold_'+str(idx+1)+'/train')
            os.makedirs(PATH_TO_DATA+'/fold_'+str(idx+1)+'/test')
        
        if len(os.listdir(PATH_TO_DATA+'/fold_'+str(idx+1)+'/train')) > 0:
            print('Already files in folder, will exit!')
            exit()

        for file_train in fold[0]:
            shutil.copy(PATH_TO_DATA+'/'+file_train+'.png', PATH_TO_DATA+'/fold_'+str(idx+1)+'/train')
            shutil.copy(PATH_TO_DATA+'/'+file_train+'.xml', PATH_TO_DATA+'/fold_'+str(idx+1)+'/train')
        for file_test in fold[1]:
            shutil.copy(PATH_TO_DATA+'/'+file_test+'.png', PATH_TO_DATA+'/fold_'+str(idx+1)+'/test')
            shutil.copy(PATH_TO_DATA+'/'+file_test+'.xml', PATH_TO_DATA+'/fold_'+str(idx+1)+'/test')

if __name__ == '__main__':
    PATH_TO_DATA = '../serialize_raw_640_3'
    copy_files(PATH_TO_DATA)