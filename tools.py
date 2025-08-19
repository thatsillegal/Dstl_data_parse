import cv2
from shapely.geometry import MultiPolygon, Polygon
import numpy as np
import os

# Scale polygons to match image:
def get_scalers(im_size,x_max,y_min):
    h,w = im_size
    w_ = w * ( w/ ( w + 1))
    h_ = h * ( h/ ( h + 1))
    return w_ / x_max, h_ / y_min

# Create a mask from polygons:
def get_polygons_list(polygons):
    if isinstance(polygons, Polygon):
        return [polygons]
    elif isinstance(polygons, MultiPolygon):
        return list(polygons.geoms)  # MultiPolygon 中的每个 Polygon
    else:
        return []

def mask_for_polygons(im_size,polygons):
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

# 保存函数（支持中文路径）
def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # 自动建目录
    ext = os.path.splitext(path)[1]  # 提取扩展名 (.png / .jpg)
    ok, encoded_img = cv2.imencode(ext, img)
    if ok:
        encoded_img.tofile(path)  # 用 tofile 写入，支持中文路径
        print(f"保存成功 -> {path}")
    else:
        print(f"保存失败 -> {path}")