# Renames every file in a source file
# Assumes that all file names are zero-padded index numbers
# i.e. 001, 002, 003, etc
import os


# NOTE: Config vars
prefix_or_suffix = True # True for prefix
name_fix = 'fucckkkkkk_'
source_path = r'C:\Users\wanga\Documents\GitHub\Bird-Cam\Dataset\Scraped Images\Scraped Images\_squirrel__climbing_bird_feeder'


# change working directory to source
os.chdir(source_path)
print('\n\n')
for _, _, filenames in os.walk(source_path):
    for file_name in filenames:
        try:
            if prefix_or_suffix:
                os.rename(os.path.join(source_path, file_name), os.path.join(source_path, name_fix+file_name))
            else:
                os.rename(os.path.join(source_path, file_name), os.path.join(source_path, file_name+name_fix))
        except FileNotFoundError:
            print(f'ERROR LOG: Could not find \'{file_name}\'')
    break
