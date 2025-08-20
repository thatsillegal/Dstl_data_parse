import os
import tools

if __name__ == '__main__':

    base_path = r"D:\OneDrive\工作\青研社课题\数据集\dstl-satellite-imagery-feature-detection"
    input_dir = os.path.join(base_path, "three_band", "output_folder", "images")

    output_dir = os.path.join(base_path, "three_band", "tiles_folder", "images")
    os.makedirs(output_dir, exist_ok=True)

    scales = [0.5,1.0,1.5,2.0]
    tools.process_folder(input_dir, output_dir,tile_size=500, scales=scales)