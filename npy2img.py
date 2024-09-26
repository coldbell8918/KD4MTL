import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Choose Image')
parser.add_argument('--number', default='0', type=str, help='image root')
opt = parser.parse_args()

# Load the arrays
img_array_depth = np.load('/home/park/MTL/KD4MTL/data/nyuv2/val/depth/'+opt.number+'.npy')
img_array_img = np.load('/home/park/MTL/KD4MTL/data/nyuv2/val/image/'+opt.number+'.npy')
img_array_label = np.load('/home/park/MTL/KD4MTL/data/nyuv2/val/label/'+opt.number+'.npy')
img_array_normal = np.load('/home/park/MTL/KD4MTL/data/nyuv2/val/normal/'+opt.number+'.npy')

# Optionally scale the data if needed
img_array_img = (img_array_img - img_array_img.min()) / (img_array_img.max() - img_array_img.min()) * 255
img_array_img = img_array_img.astype(np.uint8)

img_array_label = (img_array_label - img_array_label.min()) / (img_array_label.max() - img_array_label.min()) * 255
img_array_label = img_array_label.astype(np.uint8)

img_array_depth = (img_array_depth - img_array_depth.min()) / (img_array_depth.max() - img_array_depth.min()) * 255
img_array_depth = img_array_depth.astype(np.uint8)


img_array_normal = (img_array_normal - img_array_normal.min()) / (img_array_normal.max() - img_array_normal.min()) * 255
img_array_normal = img_array_normal.astype(np.uint8)

# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Display the images
axs[0].imshow(img_array_img)
axs[0].set_title('Image')
axs[0].axis('off')  # Hide axis

axs[1].imshow(img_array_label)
axs[1].set_title('Label')
axs[1].axis('off')  # Hide axis

axs[2].imshow(img_array_depth)
axs[2].set_title('Depth')
axs[2].axis('off')  # Hide axis

axs[3].imshow(img_array_normal)
axs[3].set_title('normal')
axs[3].axis('off')  # Hide axis

# Show the plot
plt.tight_layout()
plt.show()
