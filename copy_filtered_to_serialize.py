import configparser
import os
import re
import shutil

"""
Script mirrors the content in ../serialize_raw_640 but instead of the original
.png-file, corresponding .png-file from DESTINATION_PATH_DENOISED and 
DESTINATION_PATH_COMBINED is copied.
"""

def copy_files(SOURCE_PATH_RAW, 
    SOURCE_PATH_DENOISED, 
    SOURCE_PATH_COMBINED, 
    DESTINATION_PATH_DENOISED, 
    DESTINATION_PATH_COMBINED):
    pattern = re.compile('.*\.xml$')
    # Traverse through the folders
    for folder_fold in os.listdir(SOURCE_PATH_RAW):
        if os.path.isdir(SOURCE_PATH_RAW+'/'+folder_fold):
            for folder_train_test in os.listdir(SOURCE_PATH_RAW+'/'+folder_fold):
                for filename in os.listdir(SOURCE_PATH_RAW+'/'+folder_fold+'/'+folder_train_test):
                    if pattern.match(filename):
                        if not os.path.exists(DESTINATION_PATH_DENOISED+'/'+folder_fold+'/'+folder_train_test):
                            os.makedirs(DESTINATION_PATH_DENOISED+'/'+folder_fold+'/'+folder_train_test)
                            os.makedirs(DESTINATION_PATH_COMBINED+'/'+folder_fold+'/'+folder_train_test)

                        # Copy the files to respective folder
                        file_denoised = filename.split('.')[0]+'.png'
                        file_combined = filename.split('.')[0]+'.png'
                        shutil.copy(SOURCE_PATH_RAW+'/'+folder_fold+'/'+folder_train_test+'/'+filename, \
                            DESTINATION_PATH_DENOISED+'/'+folder_fold+'/'+folder_train_test+'/'+filename)
                        shutil.copy(SOURCE_PATH_DENOISED+'/'+file_denoised, \
                            DESTINATION_PATH_DENOISED+'/'+folder_fold+'/'+folder_train_test+'/'+file_denoised)
                        shutil.copy(SOURCE_PATH_RAW+'/'+folder_fold+'/'+folder_train_test+'/'+filename, \
                            DESTINATION_PATH_COMBINED+'/'+folder_fold+'/'+folder_train_test+'/'+filename)
                        shutil.copy(SOURCE_PATH_COMBINED+'/'+file_combined, \
                            DESTINATION_PATH_COMBINED+'/'+folder_fold+'/'+folder_train_test+'/'+file_combined)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('preprocessing.config')
    copy_filtered_to_serialize = config['COPY_FILTERED_TO_SERIALIZE']

    copy_files(
        copy_filtered_to_serialize['source_path_raw'], 
        copy_filtered_to_serialize['source_path_denoised'], 
        copy_filtered_to_serialize['source_path_combined'], 
        copy_filtered_to_serialize['destination_path_denoised'], 
        copy_filtered_to_serialize['destination_path_combined']
    )
