import os
from collections import defaultdict
import csv
import sys
import matplotlib.pyplot as plt

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

# 设置 CSV 文件中单个字段允许的最大长度（字节数）
csv.field_size_limit(2**31 - 1);

base_path = r"D:\OneDrive\工作\青研社课题\数据集\dstl-satellite-imagery-feature-detection"


IM_ID = '6120_2_2'
POLY_TYPE = '1'  # buildings

# Load grid size
x_max, y_min = None, None
csv_path = os.path.join(base_path, "grid_sizes.csv", "grid_sizes.csv")
for _im_id, _x , _y in csv.reader(open(csv_path)):
    if _im_id == IM_ID:
        x_max, y_min = float(_x), float(_y)
        break

# Load train poly with shapely
train_polygons = None
csv_path = os.path.join(base_path, "train_wkt_v4.csv", "train_wkt_v4.csv")
for _im_id , _poly_type, _poly in csv.reader(open(csv_path)):
    if _im_id == IM_ID and _poly_type == POLY_TYPE:
        train_polygons = shapely.wkt.loads(_poly)
        break

# Read image with tiff
pic_path = os.path.join(base_path,"three_band","three_band",IM_ID + ".tif")
im_rgb  = tiff.imread(pic_path).transpose([1,2,0]) #大多数图像处理库（matplotlib、OpenCV、PIL）期望的格式是：高，宽，通道
im_size = im_rgb.shape[:2] #取前两个维度 (H, W)，丢掉通道数 C。

# Scale polygons to match image:
def get_scalers():
    h,w = im_size
    w_ = w * ( w/ ( w + 1))
    h_ = h * ( h/ ( h + 1))
    return w_ / x_max, h_ / y_min

x_scalers, y_scalers = get_scalers()

if train_polygons is None:
    print("Warning: train_polygons is None, skipping scaling")
else:
    train_polygons_scaled = shapely.affinity.scale(
        train_polygons, xfact=x_scalers, yfact=y_scalers, origin=(0,0,0)
    )

# Create a mask from polygons:
def get_polygons_list(polygons):
    if isinstance(polygons, Polygon):
        return [polygons]
    elif isinstance(polygons, MultiPolygon):
        return list(polygons.geoms)  # MultiPolygon 中的每个 Polygon
    else:
        return []

def mask_for_polygons(polygons):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

polygons_list = get_polygons_list(train_polygons_scaled)
train_mask = mask_for_polygons(polygons_list)

# A helper for nicer display
def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

# Check that image and mask are aligned. Image:
pic_show = 255 * scale_percentile(im_rgb[2900:3200,2000:2300])
pic_show_uint8 = (pic_show).astype(np.uint8)
# matplotlib 的 imshow 默认对 float 数据会把 01 映射为 01，超过 1 的值就被当作“过曝”。
plt.imshow(pic_show_uint8)
plt.show()


def show_mask(m):
    mask_show = (255 * np.stack([m,m,m])
                 .transpose([1,2,0])
                 .astype(np.uint8))
    plt.imshow(mask_show)
    plt.show()

show_mask(train_mask[2900:3200, 2000:2300])