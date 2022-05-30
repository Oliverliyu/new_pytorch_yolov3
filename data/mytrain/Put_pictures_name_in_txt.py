# 用来把我们训练需要的图像地址放到一个txt文件中
# 在一个文件夹中有好多个文件夹，这些文件夹中存放着好多图片
# 把这些图片的地址放到一个txt文件中
import os
import pathlib
train_path = "all_img.txt"  # train的目标文件
valid_path = "valid.txt"
tp = open(train_path, "w")
vp = open(valid_path, "w")
paths = "E:\\datasets_Traffic_signs\\CCTSDB\\images"
paths = pathlib.PurePath(paths)
paths = pathlib.PurePath.as_posix(paths)
print(paths)
filenames = os.listdir(paths)
new_filenames = []  # 其中存放的是每一个在大的文件夹下的每一个文件夹的路径
for filename in filenames:
    new_filenames.append(pathlib.PurePosixPath(paths)/filename)
for i in range(16):  # 把每一个的文件夹下的图片地址放到一个txt文件中
    picture_dirs = os.listdir(new_filenames[i])  # 把文件夹中的图片的名字摆出来，这是一个列表
    for picture_dir in picture_dirs:
        picture_dir = pathlib.PurePosixPath(new_filenames[i])/picture_dir

        tp.write(str(picture_dir) + "\n")

for i in range(4, 6):
    picture_dirs = os.listdir(new_filenames[i])  # 把文件夹中的图片的名字摆出来，这是一个列表
    for picture_dir in picture_dirs:
        picture_dir = pathlib.PurePosixPath(new_filenames[i]) / picture_dir

        vp.write(str(picture_dir) + "\n")