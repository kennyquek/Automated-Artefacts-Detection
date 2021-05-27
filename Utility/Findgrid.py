import glob, os
from shutil import copyfile
import re


path_array_holder = []
fold = "5"
main_path = "C:/Users/Kenny/Desktop/Whole angio/validation_" + fold

sub_folder_array = os.listdir(main_path)

print(str(sub_folder_array))

not_found_array = []

for sub_folder_name in sub_folder_array:

    dir = os.path.join(main_path, sub_folder_name)
    dataset_name = os.listdir(dir)
    # print(dataset_name)


    # path = "C:/Users/kenny/Desktop/Kenny/patches collector/structural/5x5 grid " + sub_folder_name


    for file_name in dataset_name:
        path = "C:/Users/kenny/Desktop/Kenny/patches collector/structural/5x5 grid normal"

        replaced = re.sub('Angiography_Superficial', 'Structure_Whole Eye', file_name)
        dest = "C:/Users/kenny/Desktop/Whole angio/validation_" + fold + " structural/" + sub_folder_name
        if os.path.isdir(dest) == False:
            os.makedirs(dest)

        total_file_name_array = os.listdir(path)
        if replaced in total_file_name_array:
            # print("grid is found")
            index = total_file_name_array.index(replaced)
            copyfile(path + "/"+ total_file_name_array[index] , dest + "/" + replaced)
            continue

        path = "C:/Users/kenny/Desktop/Kenny/patches collector/structural/5x5 grid defects"
        total_file_name_array = os.listdir(path)
        if replaced in total_file_name_array:
            # print("grid is found")
            index = total_file_name_array.index(replaced)
            copyfile(path + "/"+ total_file_name_array[index] , dest + "/" + replaced)
            continue

        full_image_name = replaced.split(".")[0]
        not_found_array.append(full_image_name)
        print(full_image_name + " grid is not found " + replaced.split(".")[1])


if not_found_array.__len__() == 0:
    print("all found!")

print(set(not_found_array))

# Find all full images
for half_name in not_found_array:
    name = half_name + ".bmp"
    for root, dirs, files in os.walk("C:/Users/kenny/Desktop/Batch1 167-335 POAG"):
        found = False
        if name in files:
            found = True
            file_path = os.path.join(root, name)
            copyfile(file_path, "C:/Users/kenny/Desktop/collect/image" + "/" + name)

        if found == False:
            zxc = name.split("_")
            # print(str(zxc[9]))
            num = int(zxc[9])
            num += 1
            # print(str(num))
            zxc[9] = str(num)
            # combine string again
            combine = ""
            total = zxc.__len__()
            counter = 0
            for z in zxc:
                counter += 1
                if counter == total:
                    combine += (z)
                else:
                    combine += (z + "_")
            # print(combine)
            if combine in files:
                file_path = os.path.join(root, combine)
                copyfile(file_path, "C:/Users/kenny/Desktop/collect/image" + "/" + name)




