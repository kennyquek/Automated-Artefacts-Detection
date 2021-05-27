import cv2 as cv

# import glob, os
# from shutil import copyfile
# import re
#
#
# path_array_holder = []
#
# main_path = "C:/Users/Kenny/Desktop/Whole angio/validation_5 structural"
#
# sub_folder_array = os.listdir(main_path)
#
# print(str(sub_folder_array))
#
# for sub_folder_name in sub_folder_array:
#
#     dir = os.path.join(main_path, sub_folder_name)
#     dataset_name = os.listdir(dir)
#
#     for file_name in dataset_name:
#         img = cv.imread(main_path + "/" + sub_folder_name + "/" + file_name)
#         cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
#
#         dest_path = "C:/Users/Kenny/Desktop/Whole angio/validation_1 structural"
#         cv.imwrite(dest_path + "/" + sub_folder_name + "/" + file_name, img)

img = cv.imread("C:/Users/kenny/PycharmProjects/Tensorflow/venv/testt.bmp")
cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)

dest_path = "C:/Users/kenny/PycharmProjects/Tensorflow/venv/testt_norm.bmp"
cv.imwrite(dest_path, img)