import os
from shutil import copyfile


contains_1 = "_Angiography_Superficial.bmp"
contains_2 = "_Structure_Whole Eye.bmp"

path_array_holder = []

main_path = "C:/Users/User/Desktop/Kenny/defects collector"

sub_folder_array = os.listdir(main_path)

print(str(sub_folder_array))

# for folder_name in sub_folder_array:
#     path_1 = main_path + "/" + folder_name
#     path_1_OD_6mm =  path_1 + "/OD/6mm"
#     path_1_OD_12mm = path_1 + "/OD/12mm"
#
#     path_1_OS_6mm =  path_1 + "/OS/6mm"
#     path_1_OS_12mm = path_1 + "/OS/12mm"
#
#     file_1 = os.listdir(path_1_OD_6mm)
#     if file_1.__len__() != 0 :
#         for name in file_1:
#             path_array_holder.append(path_1_OD_6mm + "/" + name)
#
#     file_2 = os.listdir(path_1_OD_12mm)
#     if file_2.__len__() != 0 :
#         for name in file_2:
#             path_array_holder.append(path_1_OD_12mm + "/" + name)
#
#     file_3 = os.listdir(path_1_OS_6mm)
#     if file_3.__len__() != 0:
#         for name in file_3:
#             path_array_holder.append(path_1_OS_6mm + "/" + name)
#
#     file_4 = os.listdir(path_1_OS_12mm)
#     if file_4.__len__() != 0:
#         for name in file_4:
#             path_array_holder.append(path_1_OS_12mm + "/" + name)


#print("total number of images to label: " + str(path_array_holder.__len__()))

dst = "C:/Users/User/Desktop/Kenny/defects collector"

#print(os.path.basename(dst))

# for file_name in sub_folder_array:
#
#     if contains_1 in file_name:
#         print("yes")
#         copyfile(main_path + "/" + file_name, dst + "/angio/" + os.path.basename(file_name))
#         continue
#
#     if contains_2 in file_name:
#         print("yes")
#         copyfile(main_path + "/" + file_name, dst + "/structural/" + os.path.basename(file_name))
#         continue
#
#     print("Does not contain both")

