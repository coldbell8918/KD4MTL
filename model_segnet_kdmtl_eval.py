import torch
import numpy as np
import matplotlib.pyplot as plt
from model.segnet import SegNet

# Model and data paths
pretrained_model_path = '/home/park/MTL/KD4MTL/SegNet-KD-MTL/segnet_kdmtl_model_best.pth.tar'

# Load SegNet model
class_nb = 13
model = SegNet(type_='standard', class_nb=class_nb)
model.load_state_dict(torch.load(pretrained_model_path)['state_dict'])
model.eval()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Setup figure
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
plt.ion()  # Enable interactive mode

index = [0]

def update_figure(idx):
    # Load image and label data
    data_path = f'/home/park/MTL/KD4MTL/data/nyuv2/val/image/{idx}.npy'
    depth_path = f'/home/park/MTL/KD4MTL/data/nyuv2/val/depth/{idx}.npy'
    seg_path = f'/home/park/MTL/KD4MTL/data/nyuv2/val/label/{idx}.npy'
    normal_path = f'/home/park/MTL/KD4MTL/data/nyuv2/val/normal/{idx}.npy'
    
    image_data = np.load(data_path).astype(np.float32)
    image_data = np.transpose(image_data, (2, 0, 1))
    img_array_label = np.load(seg_path)
    img_array_depth = np.load(depth_path)
    img_array_normal = np.load(normal_path)

    image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        outputs = model(image_tensor)

    # Access predictions
    semantic_pred_tensor = outputs[0][0]
    depth_pred_tensor = outputs[0][1]
    normal_pred_tensor = outputs[0][2]

    semantic_pred = semantic_pred_tensor.argmax(1).squeeze().detach().cpu().numpy()
    depth_pred = depth_pred_tensor.squeeze().detach().cpu().numpy()
    normal_pred = normal_pred_tensor.squeeze().detach().cpu().numpy()
    normal_pred = np.transpose(normal_pred, (1, 2, 0))

    normal_viz = (normal_pred - normal_pred.min()) / (normal_pred.max() - normal_pred.min())

    # Update plots
    axes[0, 0].imshow(np.transpose(image_data, (1, 2, 0)))
    axes[0, 0].set(title='Image', xticks=[], yticks=[])

    axes[0, 1].imshow(semantic_pred)
    axes[0, 1].set(title='Semantic Segmentation', xticks=[], yticks=[])

    axes[0, 2].imshow(depth_pred)
    axes[0, 2].set(title='Depth Prediction', xticks=[], yticks=[])

    axes[0, 3].imshow(normal_viz)
    axes[0, 3].set(title='Normal Estimation', xticks=[], yticks=[])

    axes[1, 0].imshow(np.transpose(image_data, (1, 2, 0)))
    axes[1, 0].set(title='Image', xticks=[], yticks=[])

    axes[1, 1].imshow(img_array_label)
    axes[1, 1].set(title='Label', xticks=[], yticks=[])

    axes[1, 2].imshow(img_array_depth)
    axes[1, 2].set(title='Depth', xticks=[], yticks=[])

    axes[1, 3].imshow(img_array_normal)
    axes[1, 3].set(title='Normal', xticks=[], yticks=[])

    plt.draw()

# Initial plot
update_figure(index[0])

def on_key(event):
    if event.key == 'right':
        index[0] = (index[0] + 1) % 654  # Assuming there are 654 images
    elif event.key == 'left':
        index[0] = (index[0] - 1) % 654
    update_figure(index[0])

# Connect the event
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show(block=True)
