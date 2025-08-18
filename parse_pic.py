import rasterio
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
import tifffile as tiff

path_03 = r"D:\OneDrive\工作\青研社课题\数据集\dstl-satellite-imagery-feature-detection\three_band\three_band\6020_2_2.tif"
path_16 = r"D:\OneDrive\工作\青研社课题\数据集\dstl-satellite-imagery-feature-detection\sixteen_band\sixteen_band\6020_2_2_M.tif"

# P = tiff.imread(path_16)
# tiff.imshow(P)

with rasterio.open(path_03) as src:
    # 读取3个波段，假设波段1是蓝，2是绿，3是红
    r = src.read(1)
    g = src.read(2)
    b = src.read(3)

    # 归一化
    def normalize(array):
        array_min, array_max = array.min(), array.max()
        return (array - array_min) / (array_max - array_min)

    rgb = np.dstack((normalize(r), normalize(g), normalize(b)))

plt.imshow(rgb)
plt.show()