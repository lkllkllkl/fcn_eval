from PIL import Image
import os
import math
import numpy as np
import argparse
import tempfile

# 命令行解析参数
def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='../server/static/classes_result')
    parser.add_argument('--pred', type=str)
    parser.add_argument('--orig', type=str)
    parser.add_argument('--number_of_classes', type=int, default=21)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


FLAGS, unparsed = parse_args()


def get_colormap(N=256):
    def bitget(val, idx): return ((val & (1 << idx)) != 0)

    cmap = []
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3

        cmap.append((r, g, b, 255))
    return cmap

cloth_colormap = get_colormap()[:59:]
voc_colormap = get_colormap()[:21:]


voc_classes = ('background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'monitor')


cloth_classes = ('background', 'accessories', 'bag', 'belt', 'blazer', 
            'blouse', 'bodysuit', 'boots', 'bra', 'bracelet',
            'cape', 'cardigan', 'clogs', 'coat', 'dress', 
            'earrings', 'flats', 'glasses', 'gloves', 'hair', 
            'hat', 'heels', 'hoodie', 'intimate', 'jacket', 
            'jeans', 'jumper', 'leggings', 'loafers', 'necklace', 
            'panties', 'pants', 'pumps', 'purse', 'ring', 
            'romper', 'sandals', 'scarf', 'shirt', 'shoes', 
            'shorts', 'skin', 'skirt', 'sneakers','socks',
            'stockings', 'suit', 'sunglasses', 'sweater','sweatshirt',
            'swimwear', 't-shirt', 'tie', 'tights', 'top', 
            'vest', 'wallet', 'watch', 'wedges')


def get_images(origin, pred, classes = voc_classes):
    """获取分割后的图片

    orgin_img: 原图路径
    pred_img: 预测结果type ndarray
    classes: list, 类别名称
    return: 各分类分割图dict
    
    """
    if classes == voc_classes:
        colormap = voc_colormap
    else:
        colormap = cloth_colormap


    origin_img = Image.open(origin).convert('RGBA')

    class_imgs = {}
    width = pred.shape[1]
    height = pred.shape[0]
    
    for i in range(height):
        for j in range(width):
            
            color_idx = pred[i][j]
            # 跳过背景色
            if color_idx == 0:
                continue

            class_name = classes[color_idx]

            if not class_name in class_imgs.keys():
                class_img = Image.new('RGBA', (width, height), (0, 0 ,0, 0))    
                class_imgs[class_name] = class_img

            
            class_img = class_imgs[class_name]
            class_img.putpixel((j, i), origin_img.getpixel((j, i)))

    # class_imgs_without_noise = {}
    # for key, value in class_imgs.items():
    #     if not is_noise(value):
    #         class_imgs_without_noise[key] = value
    return class_imgs


def color_index(color, colormap):
    """获取color在colormap中的位置
    
    color: tuple
    colormap: list
    """
    for i in range(len(colormap)):
        if same_color(color, colormap[i]):
            return i
    return -1


def same_color(color1, color2, threshold = 45):
    """根据两个颜色距离判断是否为同一颜色
    color1: (255, 255, 255, 255)
    color2: (255, 255, 255, 255)
    threshold: 通道值相差超过该值则判断为不同颜色
    """

    diff = 0
    for i in range(len(color1) - 1):
        diff = diff + abs(color1[i] - color2[i])**2
    diff = math.sqrt(diff)

    return diff < threshold


def is_noise(img, threshold = 0.9985):
    """判断某张图片是否为噪声"""
    img_arr = np.array(img)
    per = np.sum(img_arr == 0) / (img.width * img.height * 4)
    # bg_count = img_arr.reshape(img.width * img.height, 4).tolist().count([0, 0, 0, 0])
    # per = bg_count / (img.width * img.height)
    return per > threshold

def main():
    classes = voc_classes
    if FLAGS.number_of_classes == 59:
        classes = cloth_classes
    class_imgs = get_images(FLAGS.orig, FLAGS.pred, classes)
    for class_name, img in class_imgs.items():
        output_file = tempfile.mktemp(dir=FLAGS.output, prefix=class_name + '_', suffix='.png')
        img.save(output_file)

if __name__ == '__main__':
    main()