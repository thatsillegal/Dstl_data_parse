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

def process_folder(input_dir, output_dir, tile_size=500, scales=[0.5,1.0,2.0]):
    if not os.path.exists(input_dir):
        raise Exception("文件目录不存在")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png",".tif")) and "_MASK" not in filename:
            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(input_dir, filename)
            mask_path = os.path.join(input_dir, base_name + "_MASK.png")

            if not os.path.exists(mask_path):
                print(f"未找到mask: {mask_path}, 跳过")
                continue

            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

            count = split_image_with_mask(img, mask, tile_size, scales, output_dir, base_name)
            print(f"{filename} -> {count} tiles")

def split_image_with_mask (img,mask, tile_size, scales, save_dir, base_name):
    """
    将图像缩放到多个比例，并划分为 tile_size x tile_size 的小图
    """
    h, w = img.shape[:2]
    count = 0

    for scale in scales:
        new_h, new_w = int(h * scale), int(w * scale)
        interpolation_way = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        resized_img = cv2.resize(img, (new_w, new_h),interpolation_way)
        resized_mask = cv2.resize(mask, (new_w, new_h),interpolation_way)

        for y in range(0, new_h, tile_size):
            for x in range(0, new_w, tile_size):
                tile_img = resized_img[y:y + tile_size, x:x + tile_size]
                tile_mask = resized_mask[y:y + tile_size, x:x + tile_size]

                if tile_img.shape[0] == tile_size and tile_img.shape[1] == tile_size:
                    if has_foreground(tile_mask):
                        img_path = os.path.join(save_dir, f"{base_name}_s{scale}_x{x}_y{y}.png")
                        mask_path = os.path.join(save_dir, f"{base_name}_s{scale}_x{x}_y{y}_MASK.png")
                        save_image(img_path, tile_img)
                        save_image(mask_path, tile_mask)
                        count += 1
    return count

def has_foreground(mask_tile, threshold=10):
    """
    判断mask tile中是否含有前景（白色部分）
    threshold: 白色像素最少数量
    """
    whites = np.sum(mask_tile > 127)
    if whites > threshold:
        return True
    return False