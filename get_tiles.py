import os
import tools

if __name__ == '__main__':

    base_path = r"/Users/bruno/PycharmProjects/Dstl_data_parse" # TODO 修改为自己的根目录
    input_dir = os.path.join(base_path, "output_folder", "images") # TODO 修改为自己的目录

    output_dir = os.path.join(base_path, "output_folder", "tiles") # TODO 修改为自己的目录
    os.makedirs(output_dir, exist_ok=True)

    scales = [0.5,1.0,1.5,2.0] # TODO 此处的放大倍数可以自己输入，数组数量越多，产生的图片越多
    tools.process_folder(input_dir, output_dir,tile_size=500, scales=scales)