# Merges pre-split datasets with train, test, and/or val subsets
# Renames files in one subset before moving them to another subset
# Purposely does NOT recursively move any subdirectories in subsets
import os


rename_prefix = 'valid_'
source_path = 'C:\\Users\\wanga\\Desktop\\Personal\\Machine Learning\\Sophomore AI Class\\Bird, Not Bird Data\\archive\\valid'
target_path = 'C:\\Users\\wanga\\Desktop\\Personal\\Machine Learning\\Sophomore AI Class\\Bird, Not Bird Data\\archive\\test'
subdirs_to_merge = [

]


print('\n\n')
for subdir in subdirs_to_merge:
    source_subdir = os.path.join(source_path, subdir)
    target_subdir = os.path.join(target_path, subdir)
    for _, _, filenames in os.walk(source_subdir):
        for file_name in filenames:
            os.rename(os.path.join(source_subdir, file_name), os.path.join(target_subdir, rename_prefix+file_name))
        break