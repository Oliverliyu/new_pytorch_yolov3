# 这个文件的目的是为了生成CCTSDB数据集的YOLO模型所需的labels文件
# CCTSDB中的数据形式：00000.png;527;377;555;404;warning
# YOLO模型所需的labels文件的数据形式：0 0.515 0.5 0.21694873 0.18286777
# 所以我们要做的任务有两点：
# 1.把名字转化成代号
# 2.把左上角、右下角的坐标位置转化成归一化的中心点位置以及宽高

# CCTSDB中给出的groundtruth文件的地址："E:\datasets_Traffic_signs\CCTSDB\GroundTruth\GroundTruth.txt"
# labels的目标文件夹的地址为："E:\datasets_Traffic_signs\CCTSDB\labels"

# 在制作数据集的时候，要注意一个问题，那就是image的数量，和groundtruth中的行的数量不同。
# 也就是在同一张图片中会有一个或者多个框

import json
import os
import pathlib
import cv2




def To_posix(path):  # 把windows中的文件的地址转换成posix格式
    path = pathlib.PurePath(path)
    path = pathlib.PurePath.as_posix(path)
    return path


def center_point(x1, y1, x2, y2):  # 用这个函数来获得中心点坐标
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return xc, yc, w, h


def Write_label_txt(label_list, image_path):
    image_dir = os.path.dirname(image_path)  # 图片所在的文件夹
    label_dir = "labels".join(image_dir.rsplit("images", 1))  # 这是label文件夹的地址，但是现在还没有这个地址
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)  # 创建label文件夹

    # 把label_list中的内容放到txt文件中
    label_path = pathlib.PurePosixPath(label_dir) / os.path.basename(image_path)# 先创建文件路径
    label_path = os.path.splitext((label_path))[0] + ".txt"
    lp = open(label_path, "a")
    label = " ".join(label_list)
    lp.write(label + "\n")


def GroundTruth_to_labels(groundtruth_path, labels_path, image_path):
    name2id = {'warning': "0", 'mandatory': "1", 'prohibitory': "2"}
    gt = open(groundtruth_path, mode="r")
    ip = open(image_path, mode="r")
    gt_lines = gt.readlines()  # groundtruth.txt中的所有的行，每一行都代表一个图片的groundtruth
    ip_lines = ip.readlines()  # 每一个元素都是一个图片的地址（后面还有一个\n）



    count_image = 0
    for ip_line in ip_lines:  # 通过每一个图片，来从groundtruth文件中找到对应的这个groundtruth
        print(count_image)
        count_image += 1
        ip_line = ip_line.strip()

        # 通过图片地址中的文件名，来和groundtruth的每一行的第一个元素进行匹配
        # 那怎么解决冗余的这个问题呢？用大小
        for gt_line in gt_lines:

            image_name = os.path.basename(ip_line)
            image_num = int(os.path.splitext(image_name)[0])
            gt_list = gt_line.strip().split(";")
            gt_name = gt_list[0]
            gt_num = int(os.path.splitext(gt_name)[0])
            if gt_num >= image_num:
                if gt_num == image_num:  # 确定这个groundtruth是这个image的了之后，就可以了
                    for i in range(1, 5):  # 这里将计算需要用的变量转换成浮点数
                        gt_list[i] = float(gt_list[i])
                    label_list = []  # 存放要放到label的txt文件中的每一个内容，分别是[标识号, 中心点x坐标, 中心点y坐标, 框的w, 框的y]

                    # 这里把第一个标识号设置好
                    label_list.append(name2id[gt_list[5]])

                    # 这里设置中心点坐标
                    xc, yc, w, h = center_point(gt_list[1], gt_list[2], gt_list[3], gt_list[4])


                    # 归一化
                    image = cv2.imread(ip_line)
                    image_h, image_w, _ = image.shape  # 此处注意w和h的位置
                    xc = round(xc / image_w, 6)
                    yc = round(yc / image_h, 6)
                    w = round(w / image_w, 6)
                    h = round(h / image_h, 6)

                    for i in (xc, yc, w, h):  # 把标记框和中心店坐标，放到列表中
                        label_list.append(str(i))

                    # 把列表中的标签值放到txt文件中
                    Write_label_txt(label_list, ip_line)

                else:
                    break  # 如果要是gt_num超过啦image_num那就看下一个图像



if __name__ == "__main__":
    groundtruth_path = "E:\datasets_Traffic_signs\CCTSDB\GroundTruth\GroundTruth.txt"
    groundtruth_path = To_posix(groundtruth_path)

    labels_path = "E:\datasets_Traffic_signs\CCTSDB\labels"
    labels_path = To_posix(labels_path)

    image_path = "C:\\Users\98053\Desktop\object_detection\papers\YOLOv3\PyTorch_YOLOv3\data\mytrain\\all_img.txt"
    image_path = To_posix(image_path)
    print(image_path)

    # 进入到groundtruth文件中，遍历每一个groundtruth，然后通过转化成YOLO需要的形式后，放到labels文件夹中的txt文件中
    GroundTruth_to_labels(groundtruth_path, labels_path, image_path)

