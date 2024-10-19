#param:pic_name,pic_bin,bounds
import cmath
import codecs
import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from scipy.signal import convolve2d

net_count = 0
part_categories = {
        'pmos': 'PM',
        'pmos-bulk': 'PMB',
        'nmos': 'NM',
        'nmos-bulk': 'NMB',
        'gnd': 'G',
        'vdd': 'VDD',
        'resistor': 'R',
        'voltage': 'V',
        'current': 'I',
        'cross-line-curved': 'CLC',
        'port': 'PT',
        'pnp': 'Q',
        'npn': 'Q',
        'diode': 'DIO',
        'capacitor': 'C',
        'single-end-amp': 'AMP1',
        'single-input-single-end-amp': 'AMP2',
        'switch': 'SWT',
        'voltage-lines': 'VL',
        'inductor': 'IND',
        'diff-amp': 'AMP3',
        'resistor2': 'R',
    }
def img2bin(selection_img):
  selection_arr = np.array(selection_img)
  h, w, _ = selection_arr.shape
  result = np.zeros((h, w), dtype=np.uint8)
  for r in range(h):
    for c in range(w):
      if c == 284:
          print(' ', end="")
      result[r][c] = 1 if min(selection_arr[r][c]) >= 128 else 0
      print(result[r][c], end="")
    print()
  print()
  return result

def bin2img(selection_bin):
  h, w = selection_bin.shape
  selection_arr = np.zeros((h, w, 3), dtype=np.uint8)
  for r in range(h):
    for c in range(w):
      selection_arr[r][c] = [0, 0, 0] if selection_bin[r][c] == 0 else [255, 255, 255]
  return Image.fromarray(np.array(selection_arr))

def np_save(arr, fn):
  with open(fn, 'wb') as f:
    np.save(f, arr)

def reformat_name(part_name, part_counts):
  part_name_root = 'U'
  for part_category_substr in part_categories:
    if part_category_substr in part_name:
      part_name_root = part_categories[part_category_substr]
      break
  part_counts[part_name_root] += 1
  return f'{part_name_root}{part_counts[part_name_root]}'

def reformat_names(bounds):
    part_counts = defaultdict(int)
    bound_names, name_bounds, bound_orig_names = {}, {}, {}
    for part_name in bounds:
        for i, bound in enumerate(bounds[part_name]):
            bound = tuple(bound)
            short_part_name = reformat_name(part_name, part_counts)
            bound_names[bound] = short_part_name
            name_bounds[short_part_name] = bound
            bound_orig_names[bound] = part_name
    return part_counts, bound_names, name_bounds, bound_orig_names

def calc_angle(bound, conn):
    top, left, bottom, right = bound
    r, c = conn
    center_r, center_c = (top + bottom) // 2, (left + right) // 2
    dr, dc = r - center_r, c - center_c
    polar = cmath.polar(dr + dc * 1j)
    return (cmath.pi + polar[1] * 100 // 1 / 100) / cmath.pi * 180 // 1
#广度优先
#start_r:开始的行 start_c:开始的列
#从start_r和start_c开始bfs，沿着黑色像素点走，碰到bound就表示有连接
#connected: key:bound value:(r,c)
def get_conn_bfs(page_bin, trace, visited, bound_names, start_r, start_c, connected):
    h, w = page_bin.shape
    queue = [(start_r, start_c)]
    while len(queue) > 0:
        r, c = queue.pop(0)
        if r < 0 or r >= h or c < 0 or c >= w or visited[r, c] > 0 or page_bin[r, c] == 1:
            continue
        # 访问每个未访问并且page_bin在这点为0的点
        visited[r, c], trace[r, c], in_bound = 1, 1, False
        # print(f"{r} and {c}")
        # 对每个bound判断(r,c)在不在里面
        for bound in bound_names:
            top, left, bottom, right = bound
            if r > top and r < bottom and c > left and c < right:
                connected[bound], in_bound = (r, c), True   #一个bound的两个端口相连就出问题
                # print(connected[bound])
                break
        # 不在bound里面，向四个方向延申
        if not in_bound:
            #十字交点
            if c - 10 >= 0 and page_bin[r, c - 10] == 0 and c + 10 < w and page_bin[r, c + 10] == 0 and r - 10 >= 0 and \
                    page_bin[r - 10, c] == 0 and r + 10 < h and page_bin[r + 10][c] == 0:
                visited[r, c] = 0
                #
                if visited[r, c - 10] + visited[r, c + 10] + visited[r - 10, c] + visited[r + 10, c] >= 3:
                    visited[r, c] = 1
                #只能单向搜索
                if trace[r, c - 10] > 0 and trace[r, c + 10] == 0:
                    queue.extend((r, c + 1))
                elif trace[r, c - 10] == 0 and trace[r, c + 10] > 0:
                    queue.extend((r, c - 1))
                elif trace[r - 10, c] > 0 and trace[r + 10, c] == 0:
                    queue.extend((r + 1, c))
                elif trace[r - 10, c] == 0 and trace[r + 10, c] > 0:
                    queue.extend((r - 1, c))
                continue

            queue.extend([(r + dr, c + dc) for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]])
#intersection:线网的交点
#conns: key:[bound_name][angle]  value:(线网另一端的bound_name,connected[from端],connected[to端])
def get_conn_bfs_wrapper(page_bin, visited, bound_names, r, c, conns,groups):
    connected = {}
    trace = np.zeros(page_bin.shape)
    get_conn_bfs(page_bin, trace, visited, bound_names, r, c, connected)
    group = []
    if len(connected) >= 2:
        for b1 in connected:
            angle1 = calc_angle(b1, connected[b1])
            name1 = bound_names[b1]
            elem = (name1,angle1,connected[b1])
            if elem not in group:
                group.append(elem)
        b1 = None
        for b1 in connected:
            break
        angle1 = calc_angle(b1,connected[b1])
        name1 = bound_names[b1]
        if name1 not in conns:
            conns[name1] = {}
        for b2 in connected:
            if b2 == b1:
                continue
            angle2 = calc_angle(b2,connected[b2])
            name2 = bound_names[b2]
            if name2 not in conns:
                conns[name2] = {}
            conns[name1][angle1] = (name2, connected[b1], connected[b2])
            conns[name2][angle2] = (name1, connected[b2], connected[b1])
    if len(group) > 0:
        groups.append(group)
#page_bin:图片的二进制文件
#part_counts:net的数量
#bound_names:key：？  value：net的名称
#
def get_conns(part_counts, bound_names, name_bounds, bound_orig_names,page_bin,groups):
    visited, conns = np.zeros(page_bin.shape, dtype=np.int32), {}
    for bound in bound_names:
        top, left, bottom, right = bound
        for r in range(top, bottom + 1):
            get_conn_bfs_wrapper(page_bin, visited, bound_names, r, left, conns, groups)
            get_conn_bfs_wrapper(page_bin, visited, bound_names, r, right, conns, groups)
        for c in range(left, right + 1):
            get_conn_bfs_wrapper(page_bin, visited, bound_names, top, c, conns, groups)
            get_conn_bfs_wrapper(page_bin, visited, bound_names, bottom, c, conns, groups)
    cmpt_pins = {}
    for group in groups:
        for elem in group:
            name = elem[0]
            angle = elem[1]
            pix = elem[2]
            if name not in cmpt_pins:
                cmpt_pins[name] = []
            cmpt_pins[name].append((angle,pix))
    #根据角度排序
    for name in cmpt_pins:
        pins = cmpt_pins[name]
        pins.sort(key = lambda x: x[0])
        cmpt_pins[name] = pins
    return cmpt_pins
#每个group分配一个net
def allocate_nets(groups):
    global net_count
    net_groups = {}
    for group in groups:
        net_name = ''
        for item in group:
            if 'VDD' in item[0]:
                net_name = 'vdd'
                break
        if net_name == '':
            net_name = f'n{net_count}'

        net_count = net_count + 1
        net_groups[net_name] = group
    return net_groups
#十字交点分割net
def split_cross_net(net_groups,cmpt_pins):
    for name in cmpt_pins:
        if 'CLC' not in name:
            continue
        pins = cmpt_pins[name]
        pin1,pin2,pin3,pin4 = pins[0],pins[1],pins[2],pins[3]
        pin1_net,pin2_net,pin3_net,pin4_net = '','','',''
        for net in net_groups:
            group = net_groups[net]
            for elem in group:
                if elem[0] == name and elem[1] == pin1[0] and elem[2] == pin1[1]:
                    pin1_net = net
                if elem[0] == name and elem[1] == pin2[0] and elem[2] == pin2[1]:
                    pin2_net = net
                if elem[0] == name and elem[1] == pin3[0] and elem[2] == pin3[1]:
                    pin3_net = net
                if elem[0] == name and elem[1] == pin4[0] and elem[2] == pin4[1]:
                    pin4_net = net
        group1 = net_groups[pin1_net]
        group2 = net_groups[pin2_net]
        group3 = net_groups[pin3_net]
        group4 = net_groups[pin4_net]
        group1 = group1 + group2
        group3 = group3 + group4
        net_groups[pin1_net] = group1
        net_groups[pin3_net] = group3
        del net_groups[pin2_net]
        del net_groups[pin4_net]


def pmos_order(conn_angle):
  if conn_angle > 30 and conn_angle <= 150: return 1
  if conn_angle > 150 and conn_angle <= 270: return 0
  return 2

def nmos_order(conn_angle):
  if conn_angle > 30 and conn_angle <= 150: return 1
  if conn_angle > 150 and conn_angle <= 270: return 2
  return 0

def pmos_cross_order(conn_angle):
  if conn_angle > 30 and conn_angle <= 150: return 1
  if conn_angle > 150 and conn_angle <= 240: return 0
  if conn_angle > 240 and conn_angle <= 300: return 3
  return 2

def nmos_cross_order(conn_angle):
  if conn_angle > 30 and conn_angle <= 150: return 1
  if conn_angle > 150 and conn_angle <= 240: return 2
  if conn_angle > 240 and conn_angle <= 300: return 3
  return 0

def current_order(conn_angle):
  if conn_angle > 90 and conn_angle <= 270: return 1
  return 0

def voltage_order(conn_angle):
  if conn_angle > 90 and conn_angle <= 270: return 1
  return 0


part_conn_order = {
  'pmos': pmos_order,
  'nmos': nmos_order,
  'current': current_order,
  'voltage': voltage_order,
  'npn': nmos_order,
  'pnp': pmos_order,
}

part_conn_order_priority = {
  'pmos-cross': pmos_cross_order,
  'nmos-cross': nmos_cross_order,
  'pmos-mirror-cross': pmos_cross_order,
  'nmos-mirror-cross': nmos_cross_order,
  'pmos-four': pmos_cross_order,
  'nmos-four': nmos_cross_order,
  'pmos-four-mirror': pmos_cross_order,
  'nmos-four-mirror': nmos_cross_order,
  'pnp-cross': pmos_cross_order,
  'npn-cross': nmos_cross_order,
  'pnp-mirror-cross': pmos_cross_order,
  'npn-mirror-cross': nmos_cross_order,
}

def rotate_mirror_angle(angle, rotate, mirror):
  new_angle = angle + rotate
  if mirror:
    new_angle = 360 - new_angle
  return (new_angle + 720) % 360

def merge_order(conn_angles, order_func, page_bin, bound, part_counts, rotate_angle, mirror):
  max_order = max([order_func(i) for i in range(360)]) + 1
  result = [None for _ in range(max_order)]
  for angle in conn_angles:
    order = order_func(angle)
    if result[order] is not None:
      continue
    result[order] = conn_angles[angle]

  for order in range(max_order):
    if result[order] is None:
      part_counts['N'] += 1
      created_net_name = f'N{part_counts["N"]}'
      created_pos = create_pos(page_bin, bound, order_func, order, rotate_angle, mirror)
      print('created', created_net_name, created_pos, 'for', bound)
      result[order] = (created_net_name, created_pos, created_pos)
  return result


def create_pos(page_bin, bound, order_func, order, rotate_angle, mirror):
  top, left, bottom, right = bound
  pos_candidates = []
  for r in range(top, bottom):
    pos_candidates.append((r, left))
    pos_candidates.append((r, right))
  for c in range(left, right):
    pos_candidates.append((top, c))
    pos_candidates.append((bottom, c))
  for r, c in pos_candidates:
    if page_bin[r, c] == 1: continue
    angle = calc_angle(bound, (r, c))
    angle = rotate_mirror_angle(angle, rotate_angle, mirror)
    if order_func(angle) == order:
      return (r, c)
  raise Exception(f'create pos error {bound} {order} {rotate_angle} {mirror}')


def gen_spice_netlist(page_id, page_bin, bounds):
    groups = []
    part_counts, bound_names, name_bounds, bound_orig_names = reformat_names(bounds)
    cmpt_pins = get_conns(part_counts, bound_names, name_bounds, bound_orig_names,page_bin,groups)
    net_groups = allocate_nets(groups)
    split_cross_net(net_groups, cmpt_pins)



def convert_yolo_to_bound(txt_file_path, img_width, img_height):
    """
    将 YOLO 格式的标注文件转换为边界框格式。

    参数:
    - txt_file_path: YOLO 格式标注文件的路径。
    - img_width: 图像的宽度。
    - img_height: 图像的高度。

    返回:
    - bounds: 包含所有边界框的列表，每个边界框是一个字典，包含 'class_id', 'xmin', 'ymin', 'xmax', 'ymax'。
    """
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
    bounds = {}
    with open(txt_file_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            class_name = ''
            for key in category_mapping:
                if class_id == category_mapping[key]:
                    class_name = key
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            bd = []
            if class_name in bounds:
                bd = bounds[class_name]

            xmin = int((center_x - width / 2) * img_width)
            ymin = int((center_y - height / 2) * img_height)
            xmax = int((center_x + width / 2) * img_width)
            ymax = int((center_y + height / 2) * img_height)
            bd.append([ymin, xmin, ymax, xmax])
            bounds[class_name] = bd
    return bounds

def generate_test():
    img_path = './data1/test/21.png'
    label_path = './data1/test/21.txt'
    page_img_path = './tests'
    page_bin_path = './tests'
    page_img = Image.open(img_path)

    page_bin = img2bin(page_img)
    page_img.save(f'{page_img_path}/{21}.png')
    np_save(page_bin, f'{page_bin_path}/{21}.npy')
    bounds = convert_yolo_to_bound(label_path, 640, 640)
    gen_spice_netlist(1, page_bin, bounds)
generate_test()