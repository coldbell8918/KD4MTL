import numpy as np
from PIL import Image
import os


path_dir = '/home/park/MTL/KD4MTL/png_data/cityscapes/val/depth/' # '' 안에 파일을 묶고 있는 "폴더 경로"를 쓰세요
file_list = os.listdir(path_dir)
cnt = 0
for png in file_list:
    image = Image.open(path_dir + png)
    pixel = np.array(image)
    np.save("/home/park/MTL/KD4MTL/data/cityscapes/val/depth/"+str(cnt), pixel) #저장할 '폴더 경로'를 쓰세요
    cnt = int(cnt)
    cnt += 1
