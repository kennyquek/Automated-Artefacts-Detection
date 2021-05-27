# import PIL
# from PIL import Image
#
# import os
# from shutil import copyfile
#
# angio_image_path = "C:/Users/kenny/Desktop/Kenny/defects collector/angio/"
# destination_path = "C:/Users/kenny/Desktop/Kenny/defects collector/resized angio/"
#
# sub_folder_array = os.listdir(angio_image_path)
#
# print(str(sub_folder_array))
#
# for file_name in sub_folder_array:
#     img = Image.open(angio_image_path + file_name) # image extension *.png,*.jpg
#     new_width  = 500
#     new_height = 500
#     img = img.resize((new_width, new_height), Image.ANTIALIAS)
#     img.save(destination_path + file_name)


# img = Image.open(angio_image_path) # image extension *.png,*.jpg
# new_width  = 500
# new_height = 500
# img = img.resize((new_width, new_height), Image.ANTIALIAS)
# img.save('new.png')

#read the image
# im = Image.open("C:/Users/kenny/Desktop/Flipping image data/OD.bmp")

#flip image
# out = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
# out.save('transpose-output.png')