import os
import csv
import cv2
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff
import tools

# 设置 CSV 文件中单个字段允许的最大长度（字节数）
csv.field_size_limit(2**31 - 1);
base_path = r"D:\OneDrive\工作\青研社课题\数据集\dstl-satellite-imagery-feature-detection"
POLY_TYPE = '1'  # buildings

# 获取文件夹中所有的文件并去除后缀
img_folder = os.path.join(base_path,"three_band","three_band")
imgs = os.listdir(img_folder)
IM_IDS = [os.path.splitext(img)[0] for img in imgs]

# 创建一个目标文件夹（已存在不会报错）
output_folder = os.path.join(base_path,"three_band","output_folder")
images_folder = os.path.join(output_folder,"images")
os.makedirs(output_folder,exist_ok=True)
os.makedirs(images_folder,exist_ok=True)

x_max, y_min = None, None
csv1_path = os.path.join(base_path, "grid_sizes.csv", "grid_sizes.csv")
csv2_path = os.path.join(base_path, "train_wkt_v4.csv", "train_wkt_v4.csv")

# 保存完整的图像
for IM_ID in IM_IDS:

    pic_path = os.path.join(base_path,"three_band","three_band",IM_ID + ".tif")
    image_output_path = os.path.join(images_folder,IM_ID + ".png")
    mask_output_path = os.path.join(images_folder,IM_ID + "_MASK.png")

    # Load grid size
    if_found_id = False
    for _im_id, _x , _y in csv.reader(open(csv1_path)):
        if _im_id == IM_ID:
            if_found_id = True
            x_max, y_min = float(_x), float(_y)
            break
    if not if_found_id:
        continue

    # Load train poly with shapely
    if_found_poly = False
    train_polygons = None
    for _im_id , _poly_type, _poly in csv.reader(open(csv2_path)):
        if _im_id == IM_ID and _poly_type == POLY_TYPE:
            if_found_poly = True
            train_polygons = shapely.wkt.loads(_poly)
            break
    if not if_found_poly:
        continue
    if train_polygons is None:
        print("Warning: train_polygons is None, skipping scaling")
        continue

    # Read image with tiff
    im_rgb  = tiff.imread(pic_path).transpose([1,2,0]) #大多数图像处理库（matplotlib、OpenCV、PIL）期望的格式是：高，宽，通道
    im_size = im_rgb.shape[:2] #取前两个维度 (H, W)，丢掉通道数 C。

    # Draw mask
    x_scalers, y_scalers = tools.get_scalers(im_size,x_max,y_min)
    train_polygons_scaled = shapely.affinity.scale(train_polygons, xfact=x_scalers, yfact=y_scalers, origin=(0, 0, 0))
    polygons_list = tools.get_polygons_list(train_polygons_scaled)
    train_mask = tools.mask_for_polygons(im_size,polygons_list)

    # Check that image and mask are aligned. Image:
    pic_show_uint8 = (255 * tools.scale_percentile(im_rgb)).astype(np.uint8)
    mask_show = (255 * np.stack([train_mask,train_mask,train_mask]).transpose([1,2,0])).astype(np.uint8)

    # 确保通道顺序正确
    if pic_show_uint8.ndim == 3 and pic_show_uint8.shape[2] == 3:
        pic_show_uint8 = cv2.cvtColor(pic_show_uint8, cv2.COLOR_RGB2BGR)

    # 确保 mask 是二维
    if mask_show.ndim == 3 and mask_show.shape[2] == 1:
        mask_show = np.squeeze(mask_show, axis=2)

    # 保存图像和mask
    tools.save_image(image_output_path, pic_show_uint8)
    tools.save_image(mask_output_path, mask_show)
