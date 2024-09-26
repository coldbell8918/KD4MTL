import torch
import numpy as np
import matplotlib.pyplot as plt
from model.mtan_single import SegNet
import argparse


parser = argparse.ArgumentParser(description='Single Task Learning (MTAN)')
parser.add_argument('--task', default='semantic', type=str, help='choose task: semantic, depth, normal')
opt = parser.parse_args()
# Predefined paths
pretrained_model_path = '/home/park/MTL/KD4MTL/mtan-single/mtan_single_model_task_{}_'.format(opt.task) + 'model_best.pth.tar'

# Load SegNet model
model = SegNet(task=opt.task)
model.load_state_dict(torch.load(pretrained_model_path)['state_dict'], strict=False)
model.eval()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Setup figure
fig, axes = plt.subplots(1, 3, figsize=(16, 8))
plt.ion()  # Enable interactive mode

index = [0]

def update_figure(idx, task):

    data_path = f'/home/park/MTL/KD4MTL/data/nyuv2/val/image/{idx}.npy'

    # Load image and label data
    if task == 'semantic':
        label_path = f'/home/park/MTL/KD4MTL/data/nyuv2/val/label/{idx}.npy'

    elif task == 'depth':
        label_path = f'/home/park/MTL/KD4MTL/data/nyuv2/val/depth/{idx}.npy'
    
    elif task == 'normal':
        label_path = f'/home/park/MTL/KD4MTL/data/nyuv2/val/normal/{idx}.npy'

        
    image_data = np.load(data_path)
    image_data = np.transpose(image_data, (2, 0, 1))
    img_array_label = np.load(label_path)

    image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        outputs = model(image_tensor)

    # Assuming outputs is a list, access the first tensor
    pred_tensor = outputs[0]

    if task == 'semantic':
        pred = pred_tensor.argmax(1).squeeze().detach().cpu().numpy()

    elif task == 'depth':
        pred = pred_tensor.squeeze().detach().cpu().numpy()

    elif task == 'normal':
        pred = pred_tensor.squeeze().detach().cpu().numpy()
        pred = np.transpose(pred, (1, 2, 0))

    # Update plots
    axes[0].imshow(np.transpose(image_data, (1, 2, 0)))
    axes[0].set(title='Image', xticks=[], yticks=[])

    axes[1].imshow(pred)
    axes[1].set(title=task, xticks=[], yticks=[])

    axes[2].imshow(img_array_label)
    axes[2].set_title('Label')
    axes[2].axis('off')  # Hide axis

    plt.draw()

# Initial plot
update_figure(index[0], opt.task)

def on_key(event):
    if event.key == 'right':
        index[0] = (index[0] + 1) % 654  # Forward, loop back at 654
    elif event.key == 'left':
        index[0] = (index[0] - 1) % 654  # Backward, loop back at 0
    update_figure(index[0], opt.task)

# Connect the event
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show(block=True)