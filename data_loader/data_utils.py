# -*- coding: utf-8 -*-

import math
import random
import pyclipper
import numpy as np
import cv2
from data_loader.augment import DataAugment

data_aug = DataAugment()


def check_and_validate_polys(polys, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)  # x coord not max w-1, and not min 0
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)  # y coord not max h-1, and not min 0

    validated_polys = []
    for poly in polys:
        p_area = cv2.contourArea(poly)
        if abs(p_area) < 1:
            continue
        validated_polys.append(poly)
    return np.array(validated_polys)

def unshrink_offset(poly,ratio):
    area = cv2.contourArea(poly)
    peri = cv2.arcLength(poly, True)
    a = 8
    b = peri - 4
    c = 1-0.5 * peri - area/ratio
    return quadratic(a,b,c)

def quadratic(a, b, c):
    if (b * b - 4 * a * c) < 0:
        return 'None'
    Delte = math.sqrt(b * b - 4 * a * c)
    if Delte > 0:
        x = (- b + Delte) / (2 * a)
        y = (- b - Delte) / (2 * a)
        return x, y
    else:
        x = (- b) / (2 * a)
        return x

def dist_to_poly_edge(xs, ys, point):
    """计算点到多边形各边的最小距离"""
    num_points = len(xs)
    min_dist = float('inf')
    for i in range(num_points):
        j = (i + 1) % num_points
        # 点到线段的距离
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[j], ys[j]
        px, py = point

        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            dist = np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        else:
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
        min_dist = min(min_dist, dist)
    return min_dist


def generate_rbox(im_size, text_polys, text_tags, training_mask, shrink_ratio,
                  thresh_min=0.3, thresh_max=0.7):
    """
    生成shrink map (概率图GT)、threshold map (基于距离的渐变图) 和 dilated mask
    与官方DBNet实现一致，threshold map 在收缩与膨胀区域之间基于距离生成渐变值
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param text_tags: 标注文本框是否参与训练
    :param training_mask: 忽略标注为 DO NOT CARE 的矩阵
    :param shrink_ratio: 收缩比例
    :param thresh_min: threshold map 最小值
    :param thresh_max: threshold map 最大值
    :return: shrink_map, threshold_map, threshold_mask (G_d), training_mask
    """
    h, w = im_size
    shrink_map = np.zeros((h, w), dtype=np.float32)
    threshold_map = np.zeros((h, w), dtype=np.float32)
    threshold_mask = np.zeros((h, w), dtype=np.float32)

    for i, (poly, tag) in enumerate(zip(text_polys, text_tags)):
        try:
            poly = poly.astype(np.int32)
            area = cv2.contourArea(poly)
            peri = cv2.arcLength(poly, True)
            if peri == 0 or area == 0:
                continue

            # 收缩偏移量 D (与论文公式一致: D = A*(1-r^2)/L)
            D = area * (1 - shrink_ratio * shrink_ratio) / peri + 0.5
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

            # 收缩多边形 -> probability map GT
            shrinked_poly = np.array(pco.Execute(-D))
            if len(shrinked_poly) == 0:
                continue
            cv2.fillPoly(shrink_map, shrinked_poly, 1)

            if not tag:
                cv2.fillPoly(training_mask, shrinked_poly, 0)

            # 膨胀多边形 -> threshold map 的计算区域
            pco2 = pyclipper.PyclipperOffset()
            pco2.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            dilated_poly = np.array(pco2.Execute(D))
            if len(dilated_poly) == 0:
                continue

            # 标记 threshold mask (dilated text region)
            cv2.fillPoly(threshold_mask, dilated_poly, 1)

            # 在 dilated 区域内生成基于距离的 threshold map
            # 计算 bounding box 以限制搜索范围
            dilated_pts = dilated_poly[0]
            xmin = max(0, np.min(dilated_pts[:, 0]))
            xmax = min(w, np.max(dilated_pts[:, 0]))
            ymin = max(0, np.min(dilated_pts[:, 1]))
            ymax = min(h, np.max(dilated_pts[:, 1]))

            poly_xs = poly[:, 0].tolist()
            poly_ys = poly[:, 1].tolist()

            for y in range(ymin, ymax):
                for x in range(xmin, xmax):
                    if threshold_mask[y, x] == 0:
                        continue
                    dist = dist_to_poly_edge(poly_xs, poly_ys, (x, y))
                    # 将距离映射到 [thresh_min, thresh_max]
                    # 距离越近边界值越大，距离越远值越小
                    val = thresh_max - min(dist / D, 1.0) * (thresh_max - thresh_min)
                    threshold_map[y, x] = max(threshold_map[y, x], val)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print('generate_rbox error for poly:', poly)

    return shrink_map, threshold_map, threshold_mask, training_mask


def augmentation(im: np.ndarray, text_polys: np.ndarray, scales: np.ndarray, degrees: int) -> tuple:
    # the images are rescaled with ratio {0.5, 1.0, 2.0, 3.0} randomly
    im, text_polys = data_aug.random_scale(im, text_polys, scales)
    # the images are horizontally fliped and rotated in range [−10◦, 10◦] randomly
    if random.random() < 0.5:
        im, text_polys = data_aug.horizontal_flip(im, text_polys)
    if random.random() < 0.5:
        im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)
    return im, text_polys


def image_label(im: np.ndarray, text_polys: np.ndarray, text_tags: list, input_size: int = 640,
                shrink_ratio: float = 0.4, degrees: int = 10,
                scales: tuple = (0.5, 1.0, 2.0, 3.0)) -> tuple:
    """
    读取图片并生成label
    :param im: 图片
    :param text_polys: 文本标注框
    :param text_tags: 是否忽略文本的标致：true 忽略, false 不忽略
    :param input_size: 输出图像的尺寸
    :param shrink_ratio: gt收缩的比例
    :param degrees: 随机旋转的角度
    :param scales: 随机缩放的尺度 (扩充为 {0.5, 1.0, 2.0, 3.0})
    :return:
    """
    h, w, _ = im.shape
    # 检查越界
    text_polys = check_and_validate_polys(text_polys, (h, w))
    im, text_polys = augmentation(im, text_polys, scales, degrees)

    h, w, _ = im.shape
    short_edge = min(h, w)
    if short_edge < input_size:
        # 保证短边 >= inputsize
        scale = input_size / short_edge
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        text_polys *= scale

    h, w, _ = im.shape
    training_mask = np.ones((h, w), dtype=np.uint8)

    # 生成 shrink_map (概率图GT), threshold_map (距离渐变图), threshold_mask (膨胀区域)
    shrink_map, threshold_map, threshold_mask, training_mask = generate_rbox(
        (h, w), text_polys, text_tags, training_mask, shrink_ratio
    )

    score_maps = np.array([shrink_map, threshold_map], dtype=np.float32)

    imgs = data_aug.random_crop([im, score_maps.transpose((1, 2, 0)), training_mask, threshold_mask],
                                (input_size, input_size))

    # 返回全分辨率的 labels (模型使用 deconv 输出全分辨率, 不再需要 1/4 下采样)
    return imgs[0], imgs[1].transpose((2, 0, 1)), imgs[2], imgs[3]

if __name__ == '__main__':
    poly = np.array([377,117,463,117,465,130,378,130]).reshape(-1,2)
    print(poly)
    print(poly.shape) #(4,2)
    shrink_ratio = 0.5
    d_i = cv2.contourArea(poly) * (1 - shrink_ratio) / cv2.arcLength(poly, True) + 0.5
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked_poly = np.array(pco.Execute(-d_i))
    print(d_i)
    print(cv2.contourArea(shrinked_poly.astype(int)) / cv2.contourArea(poly))
    print(unshrink_offset(shrinked_poly,shrink_ratio))
