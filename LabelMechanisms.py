import os
from os import listdir
from os.path import isfile, join

import numpy as np
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image, ImageChops

from functools import partial

ROW = 5
COL = 5

color_label_red = "red"
color_label_green = "green"

job_path = 'C:/Users/Kenny/Desktop/Kenny/patches collector/structural/'
buttons_array = []
test = []

selected_grid_number = {}

path_array_holder = []
last_index = 0

def saveButtonClick():

    global cache_image
    global last_index
    global canvas
    global img
    #global image_on_canvas

    print("saved")
    print("selected_grid_number: " + str(selected_grid_number))
    image_data = np.asarray(cache_image)

    number_of_pictures = COL * ROW

    image_holder = []
    grid_width = int(image_width / COL)
    grid_height = int(image_height / ROW)

    for i in range(0, number_of_pictures):
        image_holder.append([])
        for _ in range(grid_height):
            image_holder[i].append([])


    #print(len(image_holder[0]))

    photo_number = 0
    for row_num in range(0, ROW):
        for y in range(0, grid_height):
            for col_num in range(0, COL):
                for i in range(grid_width * col_num , (grid_width * col_num + grid_width)):

                    # print("col_num " + str(col_num))
                    # print("y: " + str(y))
                    # print("i:" + str(i))
                    image_holder[photo_number + col_num][y].append(image_data[row_num * grid_height + y][i])
        photo_number += COL

    #save each photo with name by number grid, save into 2 folders (defects, normal)
    grids = np.arange(number_of_pictures)
    for g in grids:
        if g in selected_grid_number:
            myarray = np.asarray(image_holder[g])
            new_image = Image.fromarray(myarray)
            new_image.save( job_path + "5x5 grid defects/" + path_array_holder[last_index]  + "_grid" + str(g) + ".bmp", "bmp")

        else:
            myarray = np.asarray(image_holder[g])
            new_image = Image.fromarray(myarray)
            new_image.save( job_path + "5x5 grid normal/" + path_array_holder[last_index]  + "_grid" + str(g) + ".bmp", "bmp")

    #save a text file for last index

    last_index += 1

    #clear the text file
    open("last_index.txt", "w").close()

    text_file = open("last_index.txt", "a")
    text_file.write(str(last_index))
    text_file.close()

    #refresh UI
    selected_grid_number.clear()
    for btn in buttons_array:
        btn["bg"] = color_label_red

    print("opening " + main_path + "/" + path_array_holder[last_index])
    cache_image = Image.open(main_path + "/" + path_array_holder[last_index])
    img = ImageTk.PhotoImage(cache_image)

    canvas.itemconfig(image_on_canvas, image = img)

def grid_click(grid_number):
    print(grid_number)
    btn = buttons_array[grid_number]

    if btn["bg"] == color_label_red:
        selected_grid_number[grid_number] = "selected"
        btn["bg"] = color_label_green
    else:
        selected_grid_number.pop(grid_number)
        btn["bg"] = color_label_red


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    #Bounding box given as a 4-tuple defining the left, upper, right, and lower pixel coordinates.
    #If the image is completely empty, this method returns None.
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def create_grid(event=None):
    w = canvas.winfo_width()  # Get current width of canvas
    h = canvas.winfo_height()  # Get current height of canvas
    canvas.delete('grid_line')  # Will only remove the grid_line

    # Creates all vertical lines at intevals of 100
    for i in range(0, w, 100):
        canvas.create_line([(i, 0), (i, h)], tag='grid_line')

    # Creates all horizontal lines at intevals of 100
    for i in range(0, h, 100):
        canvas.create_line([(0, i), (w, i)], tag='grid_line')



file = open("last_index.txt", "r+")
last_index = int(file.read())
print(str(last_index))
# store all the job paths
path_array_holder = []

main_path = "C:/Users/kenny/Desktop/collect/image"

path_array_holder = os.listdir(main_path)

print("total number of images to label: " + str(path_array_holder.__len__()))
# for path in path_array_holder:
#     print(str(path))

root = tk.Tk()
cache_image = Image.open(main_path + "/" + path_array_holder[last_index])
img = ImageTk.PhotoImage(cache_image)

image_width = img.width()
image_height = img.height()

window_width = image_width * 2
geo = str(window_width) + "x" + str(image_height + 50) + "+0+0"

root.geometry(geo)
canvas = tk.Canvas(root, height=image_height, width=image_width)

canvas.pack(fill=tk.BOTH, expand=True)
canvas.bind('<Configure>', create_grid)
image_on_canvas = canvas.create_image(0, 0, image=img, anchor=NW)

topFrame = Frame(root)
topFrame.pack()
bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM)

save_button = Button(topFrame, text='SAVE', fg='green', command=lambda: saveButtonClick())
info = Label(topFrame, text='number of photos left ', fg='black')

save_button.grid(column=2, row=1)
info.grid(column=1, row=1)

# assign buttons to each grid
number_of_buttons = COL * ROW

button_width = int(image_width / ROW)
button_height = int(image_height / COL)

button_size = 4
grid_number = 0
y = 0
off_set = button_size * 2
for index_y in range(0, ROW):
    x = image_width
    for index_x in range(0, COL):
        # print(x)
        # print(y)
        button = Button(root, text=str(grid_number), width=button_size, height=button_size, bg='red',
                        command=partial(grid_click, grid_number), borderwidth=0)
        button.place(x=x + off_set, y=y + off_set)
        x += button_width
        grid_number += 1
        # button.pack()
        test.append(grid_number)

        buttons_array.append(button)

    y += button_height

root.mainloop()
