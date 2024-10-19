import math
import os
import shutil
import random
from PIL import Image
import cv2
import numpy as np

def classify():
    # 设置路径
    source_dir = 'D:\\forCode\\6th_integrated_circuit_eda_elite_challenge_question10_dataset-main\\all_images\\'  # 源文件夹路径
    train_dir = 'D:\\forCode\\eda\\train'  # 训练集文件夹路径
    test_dir = 'D:\\forCode\\eda\\test'  # 测试集文件夹路径

    # 创建训练集和测试集文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取所有图片文件
    all_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    json_files = {f.replace('.png', '.json') for f in all_files}  # 假设图片和JSON同名
    # 打乱文件顺序
    random.shuffle(all_files)

    # 计算训练集和测试集的数量
    train_size = int(0.8 * len(all_files))  # 80% 用于训练
    train_files = all_files[:train_size]
    test_files = all_files[train_size:]

    # 移动文件到相应的文件夹
    for file in train_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(train_dir, file))
        # 移动对应的JSON文件
        json_file = file.replace('.png', '.json')
        if json_file in json_files:
            shutil.move(os.path.join(source_dir, json_file), os.path.join(train_dir, json_file))

    for file in test_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(test_dir, file))
        # 移动对应的JSON文件
        json_file = file.replace('.png', '.json')
        if json_file in json_files:
            shutil.move(os.path.join(source_dir, json_file), os.path.join(test_dir, json_file))

    print(f'Training set size: {len(train_files)}')
    print(f'Test set size: {len(test_files)}')

def JsonToYolo():
    import json
    import os

    # 设置路径
    json_folder = 'D:\\forCode\\eda\\test'  # JSON 文件夹路径
    yolo_folder = 'D:\\forCode\\eda\\test'  # 输出 YOLO 格式的文件夹路径


    # 处理每个 JSON 文件
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)

            # 读取 JSON 文件
            with open(json_path, 'r') as f:
                data = json.load(f)
            shapes = data['shapes']

            # 获取图像宽度和高度（假设在 JSON 中有）
            image_width = data['imageWidth']
            image_height = data['imageHeight']

            # YOLO 格式的标注
            yolo_annotations = []

            for annotation in shapes:
                category_id = annotation['label']
                x1 = math.floor(annotation['points'][0][0])
                y1 = math.floor(annotation['points'][0][1])
                x2 = math.ceil(annotation['points'][1][0])
                y2 = math.ceil(annotation['points'][1][1])
                xmin = x1
                ymin = y1
                width = x2 - x1
                height = y2 - y1
                # 计算中心点和归一化宽高
                x_center = (xmin + width / 2) / image_width
                y_center = (ymin + height / 2) / image_height
                norm_width = width / image_width
                norm_height = height / image_height

                # 添加到 YOLO 格式列表
                yolo_annotations.append(f"{category_id} {x_center} {y_center} {norm_width} {norm_height}")

            # 将 YOLO 格式写入文件
            yolo_file_name = json_file.replace('.json', '.txt')  # 将 .json 替换为 .txt
            yolo_file_path = os.path.join(yolo_folder, yolo_file_name)

            with open(yolo_file_path, 'w') as yolo_file:
                for annotation in yolo_annotations:
                    yolo_file.write(annotation + '\n')

    print("转换完成。")
def deleteFilesInDirs():
    folder_path = 'D:\\forCode\\eda\\test'

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):  # 检查文件扩展名是否为 .json
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)  # 删除文件
            print(f"Deleted: {file_path}")

    print("所有 JSON 文件已删除。")
def replace():
    # 设置文件路径
    dir_path = 'D:\\forCode\\eda\\test'
    dic = {'PMOS_normal': 0, 'PMOS_cross': 1, 'PMOS_bulk': 2, 'NMOS_normal': 3, 'NMOS_cross': 4, 'NMOS_bulk': 5,
           'Voltage_1': 6, 'Voltage_2': 7,
           'current': 8, 'BJT_NPN': 9, 'BJT_NPN_cross': 10, 'BJT_PNP': 11, 'BJT_PNP_cross': 12, 'diode': 13,
           'Diso_amp': 14, 'Siso_amp': 15, 'Dido_amp': 16,
           'Capacitor': 17, 'gnd': 18, 'inductor': 19, 'Resistor_1': 20, 'Resistor_2': 21}  # 类别名称
    for file_path in os.listdir(dir_path):
        file_path = os.path.join(dir_path,file_path)
        if file_path.endswith('.png'):
            continue
        # 读取文件并修改每一行
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 修改每一行的第一个元素
        modified_lines = []
        for line in lines:
            elements = line.split()  # 将行按空格分隔
            new_value = dic.get(elements[0])  # 你想替换第一个元素的新值
            if elements and new_value is not None:  # 确保行不为空
                elements[0] = new_value  # 修改第一个元素
                modified_line = ' '.join(elements)  # 将元素重新组合成一行
                modified_lines.append(modified_line)

        # 将修改后的内容写回文件
        with open(file_path, 'w') as file:
            file.write('\n'.join(modified_lines))

import glob
def read_txt_files():
    directory = 'D:\\forCode\\eda\\test'
    ls = []
    file_ind = []
    # 查找指定目录下的所有txt文件
    txt_files = glob.glob(os.path.join(directory, '*.txt'))

    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                # 去掉行末的换行符并按空格划分
                words = line.strip().split()
                if words[0] not in ls:
                    ls.append(words[0])
                    file_name = file.split('\\')[-1]
                    file_ind.append(file_name)


    for i in range(len(ls)):
        print(ls[i])
        print(file_ind[i])

import shutil

def copy_files_and_process_txt():
    src_dir = 'D:\\forCode\\eda\\test'
    dest_dir = 'C:\\Users\\10844\\PycharmProjects\\edaCompete\\data\\test'
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 遍历源目录中的所有文件和文件夹
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)

        # 如果是文件，则复制到目标目录
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)
            # 检查文件是否是txt文件
            if src_path.endswith('.txt'):
                process_txt_file(dest_path)
#统一器件名
def process_txt_file():
    label_folder = './data1/test'  # 标注文件夹路径

    # 处理每个图像和对应的标注文件
    for filename in os.listdir(label_folder):
        if filename.endswith(('.txt')):
            # 读取文件内容
            label_path = os.path.join(label_folder, filename)
            with open(label_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 修改每行的第一个元素
            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                modified_parts = parts
                if parts:  # 确保行不为空
                    if 'pmos' in parts[0]:
                        modified_parts = ['pmos'] + parts[1:]
                    elif 'nmos' in parts[0]:
                        modified_parts = ['nmos'] + parts[1:]
                    elif 'pnp' in parts[0]:
                        modified_parts = ['pnp'] + parts[1:]
                    elif 'npn' in parts[0]:
                        modified_parts = ['npn'] + parts[1:]
                    elif 'capacitor' in parts[0]:
                        modified_parts = ['capacitor'] + parts[1:]
                    elif 'inductor' in parts[0]:
                        modified_parts = ['inductor'] + parts[1:]
                    elif 'switch' in parts[0]:
                        modified_parts = ['switch'] + parts[1:]
                    elif 'resistor2' in parts[0]:
                        modified_parts = ['resistor2'] + parts[1:]
                    # 假设我们要将第一个元素改为"Modified"
                    modified_lines.append(" ".join(modified_parts))
                else:
                    modified_lines.append(line)
            os.remove(label_path)
            # 将修改后的内容写回文件
            with open(label_path, 'w', encoding='utf-8') as file:
                file.write("\n".join(modified_lines))

#检查调整后label是否正确
def check_label():
    # 设置路径
    image_folder = './tests'  # 图像文件夹路径
    label_folder = './tests'  # YOLO 标注文件夹路径

    # 显示每个标注的类别颜色
    colors = {
        '0': (255, 0, 0),  # 类别 0 的边框颜色为红色
        '1': (0, 255, 0),  # 类别 1 的边框颜色为绿色
        '2': (0, 0, 255),  # 类别 2 的边框颜色为蓝色
        # 可以根据需要添加更多类别和对应颜色
    }

    # 处理每个图像和对应的标注文件
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 检查文件类型
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, filename.replace('.png', '.txt'))  # 假设标注文件为.txt

            # 读取图像
            img = cv2.imread(image_path)
            # 获取图像的尺寸
            height, width, _ = img.shape
            print(f"{height} and {width}")
            # 读取标注文件并绘制框
            with open(label_path, 'r') as label_file:
                lines = label_file.readlines()

            for line in lines:
                elements = line.split()
                if len(elements) >= 5:  # 检查格式是否正确
                    category_id = elements[0]  # 类别 ID
                    x_center = float(elements[1]) * width  # 中心点 x
                    y_center = float(elements[2]) * height  # 中心点 y
                    box_width = float(elements[3]) * width  # 宽度
                    box_height = float(elements[4]) * height  # 高度

                    # 计算边界框的坐标
                    xmin = int(x_center - box_width / 2)
                    ymin = int(y_center - box_height / 2)
                    xmax = int(x_center + box_width / 2)
                    ymax = int(y_center + box_height / 2)
                    print(f"{xmin},{ymin},{xmax},{ymax}")
                    # 绘制边界框
                    color = (255, 0, 0)  # 默认白色
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(img, f'ID: {category_id}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 显示图像
            cv2.imshow('Image with Bounding Boxes', img)
            cv2.waitKey(0)  # 等待按键
            cv2.destroyAllWindows()  # 关闭窗口

#调整图片尺寸并调整label
def resize_image_with_padding( new_size=(640, 640), background_color=(255, 255, 255)):
    image_folder = './data/test'  # 图像文件夹路径
    label_folder = './data/test'  # YOLO 标注文件夹路径
    output_path = './data1/test'
    os.makedirs(output_path, exist_ok=True)
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 检查文件类型
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, filename.replace('.png', '.txt'))
            original_image = Image.open(image_path)
            # 计算原始图片的宽高
            original_width, original_height = original_image.size

            # 计算缩放比例
            scale_factor = min(new_size[0] / original_width, new_size[1] / original_height)

            # 计算调整尺寸后的图片宽高
            adj_width = int(original_width * scale_factor)
            adj_height = int(original_height * scale_factor)

            # 调整图片尺寸
            resized_image = original_image.resize((adj_width, adj_height), Image.LANCZOS)

            # 创建新画布，背景填充为白色
            padded_image = Image.new('RGB', new_size, background_color)

            # 计算粘贴位置（使图片居中）
            left = (new_size[0] - adj_width) // 2
            top = (new_size[1] - adj_height) // 2

            # 将调整后的图片粘贴到画布上
            padded_image.paste(resized_image, (left, top))

            with open(label_path, 'r') as label_file:
                lines = label_file.readlines()
            new_labels = []
            for line in lines:
                elements = line.split()
                if len(elements) >= 5:  # 检查格式是否正确
                    category_id = elements[0]  # 类别 ID
                    x_center = float(elements[1]) * original_width  # 原始图像中心点 x
                    y_center = float(elements[2]) * original_height  # 原始图像中心点 y
                    width = float(elements[3]) * original_width  # 原始图像宽度
                    height = float(elements[4]) * original_height  # 原始图像高度

                    # 计算新的归一化坐标
                    new_x_center = (float(elements[1]) * adj_width + left) / 640
                    new_y_center = (float(elements[2]) * adj_height + top) / 640
                    new_width = width * scale_factor / 640
                    new_height = height * scale_factor / 640

                    new_labels.append(f"{category_id} {new_x_center} {new_y_center} {new_width} {new_height}\n")
            new_label_path = os.path.join(output_path, filename.replace('.png', '.txt'))
            with open(new_label_path, 'w') as new_label_file:
                new_label_file.writelines(new_labels)
            # 保存或显示图片
            padded_image.save(os.path.join(output_path, filename))
            print(filename)

def transfer_category_to_id():
    category_mapping = {
        'pmos': 0,
        'pmos-bulk': 1,
        'nmos': 2,
        'nmos-bulk': 3,
        'gnd': 4,
        'vdd': 5,
        'resistor': 6,
        'voltage': 7,
        'current': 8,
        'cross-line-curved': 9,
        'port': 10,
        'pnp': 11,
        'npn': 12,
        'diode': 13,
        'capacitor': 14,
        'single-end-amp': 15,
        'single-input-single-end-amp': 16,
        'switch': 17,
        'voltage-lines': 18,
        'inductor': 19,
        'diff-amp': 20,
        'resistor2': 21,
    }
    # 设置路径
    label_folder = './data1/test'  # 原始标注文件夹路径
    output_label_folder = './data1/test'  # 输出标注文件夹路径

    # 创建输出文件夹
    os.makedirs(output_label_folder, exist_ok=True)

    # 处理每个标注文件
    for filename in os.listdir(label_folder):
        if filename.endswith('.txt'):  # 只处理 .txt 文件
            label_path = os.path.join(label_folder, filename)

            # 读取原始标注文件
            with open(label_path, 'r') as label_file:
                lines = label_file.readlines()

            # 转换标注内容
            new_labels = []
            for line in lines:
                elements = line.split()
                if len(elements) >= 5:  # 检查格式是否正确
                    object_name = elements[0]  # 物体名称
                    if object_name[0] >= '0' and object_name[0] <= '9':
                        new_labels.append(line)
                        continue
                    if object_name in category_mapping:
                        category_id = category_mapping[object_name]  # 获取对应的 ID
                        # 重新构建行，ID 在前，保留其他信息
                        new_line = f"{category_id} {' '.join(elements[1:])}\n"
                        new_labels.append(new_line)
                    else:
                        print(filename)
                        print(f"Warning: '{object_name}' not found in mapping.")
                        #new_labels.append(line)

            # 保存新的标注文件
            os.remove(label_path)
            new_label_path = os.path.join(output_label_folder, filename)
            with open(new_label_path, 'w') as new_label_file:
                new_label_file.writelines(new_labels)

    print("标注文件已转换为数字 ID。")


def normalize_image():
    dir_path = './data1/train'
    output_path = dir_path
    for fileName in os.listdir(dir_path):
        if fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dir_path,fileName)
            # 打开图片
            with Image.open(image_path) as img:
                # 将图片转换为浮点数类型，以便进行归一化
                img = img.convert('RGB')  # 确保图片是RGB模式
                img = img.point(lambda p: p / 255.0)
                # 保存归一化后的图片
                img.save(image_path)
def convert_pic():
    dir_path = './data1/train'
    for fileName in os.listdir(dir_path):
        if fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 读取图片
            image_path = os.path.join(dir_path,fileName)
            img = cv2.imread(image_path)

            # 转换为灰度图
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 二值化处理，设置阈值为1（可以根据需要调整）
            _, binary_img = cv2.threshold(gray_img, 1, 1, cv2.THRESH_BINARY)

            # 将二进制图像转换为0和1的数组
            binary_array = np.where(binary_img == 1, 1, 0)

            # 保存结果
            output_path = image_path  # 替换为你的输出图片路径
            cv2.imwrite(output_path, binary_array.astype(np.uint8) * 255)

            print("图像处理完成，已保存为：", output_path)
# 使用示例
check_label()