import os
from shutil import copyfile
import random

fold = 5

destination = "C:/Users/Kenny/Desktop/Whole structural"
main_path = "C:/Users/Kenny/Desktop/Whole structural/train"

sub_folder_array = os.listdir(main_path)

print(str(sub_folder_array))

for sub_folder_name in sub_folder_array:

    dir = os.path.join(main_path, sub_folder_name)
    dataset_name = os.listdir(dir)

    # Shuffle the dataset
    random.shuffle(dataset_name)

    num_of_data = len(dataset_name)
    # print(num_of_data)
    number_of_validation_set = num_of_data//fold

    print(sub_folder_name + ": " + str(number_of_validation_set))


    # split data using original training sets for fold times (5)
    for k in range(1, (fold + 1)):
        copying_counter = 0
        min =  (k - 1) * number_of_validation_set
        max = min + number_of_validation_set

        dest_v = destination + "/validation_" + str(k) + "/" + sub_folder_name
        if not os.path.exists(dest_v):
            os.makedirs(dest_v)

        dest_t = destination + "/train_" + str(k) + "/" + sub_folder_name
        if not os.path.exists(dest_t):
            os.makedirs(dest_t)

        for name in dataset_name:
            copying_counter += 1
            print(copying_counter)
            file_path = os.path.join(dir, name)

            if copying_counter > min and copying_counter <= max:
                # copy file to new train folder
                # print("copy file to validation set to " + str(sub_folder_name) + " fold " + str(k))
                copyfile(file_path, dest_v + "/" + name)
            else:
                # print("copy file to train set " + str(sub_folder_name))
                copyfile(file_path, dest_t + "/" + name)

