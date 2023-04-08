import os
import cv2
import torch
import numpy as np
from PIL import Image
import random
import torchvision
from torchvision.transforms import ToTensor, Compose, Resize
from copy import deepcopy
xmin_at_y_0 = 0.05
xmax_at_y_0 = 0.25

s = [0.35, 0.45, 0.55, 0.65, 0.75]

x_turb_mu = 0.05
x_turb_sigma = 0.02
def sample_darkening_params(index):
    """
        Sample two points, and generate its slope
    """

    b = random.uniform(xmin_at_y_0, xmax_at_y_0)
    print(index, s[index])
    a = np.random.normal(s[index], 0.02)  # Assume slope = const. for all channels

    return a, b
def darken(x: torch.Tensor, index):
    a, b = sample_darkening_params(index)
    # R and G are usually larger than G and R, respectively.
    # y_g = a * x + b
    while True:

        x_G = b
        y_G = a * (1 - b)
        mu, sigma = x_turb_mu, x_turb_sigma
        x_R = x_G + np.random.normal(mu, sigma)
        x_B = x_G - np.random.normal(mu, sigma)
        y_R = y_G - np.random.normal(mu, sigma)
        y_B = y_G + np.random.normal(mu, sigma)

        if ((x_B < x_G < x_R) and (y_R < y_G < y_B)) :
            print(x_B,x_G,x_R, y_R, y_G, y_B)
            break

    a_list = [y_R / (1 - x_R), a, y_B / (1 - x_B)]
    b_list = [x_R, x_G, x_B]

    #b = torch.Tensor(b).view(3, 1, 1)
    #b = b.repeat(1, x.size(1), x.size(2))
    for i in range(3) :
        print(index, a_list[i], b_list[i])
        x[i] = a_list[i] * (x[i] - b_list[i])
    x = torch.clamp(x, min=0.0, max=1.0)
    return x


train_data_dir="./"
shadowfree_imgs_dir = os.path.join(train_data_dir, 'shadow_free_rename') ##./dataset/its/'B'
matte_imgs_dir = os.path.join(train_data_dir, 'matte_rename')

save_dir = train_data_dir + '/shadow'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
i = 0
n =len(os.listdir(matte_imgs_dir))
for file_name in os.listdir(matte_imgs_dir):
    mask_name = file_name
    shadow_free_img_name = mask_name.split('_')[0] + '.jpg'
    index = mask_name.split('_')[-1].split('.')[0]
    #print(index)
    #print(mask_name, shadow_free_img_name)
    target_pil = Image.open(os.path.join(shadowfree_imgs_dir, shadow_free_img_name)).convert('RGB')
    mask_pil = Image.open(os.path.join(matte_imgs_dir, mask_name)).convert('L')
    transformer = Compose([ToTensor()])
    #transformer_mask = Compose([Resize(size=(target_pil.size[1], target_pil.size[0])), ToTensor()])
    target_tensor = transformer(target_pil)
    mask_tensor = transformer(mask_pil)
    #print(target_tensor.size(), mask_tensor.size())
    x = deepcopy(target_tensor)
    target_dark = darken(target_tensor, int(index))
    input_tensor = mask_tensor * target_dark
    input_tensor += (1 - mask_tensor) * x

    input_image = input_tensor
    torchvision.utils.save_image(input_image.cpu(), os.path.join(save_dir, mask_name))
    i += 1
    print(str(i) + '/' + str(n))
    '''
    # print(input_image.size())
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data

        image_numpy = image_tensor[:].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')
        image_numpy = np.transpose(image_numpy,
                                   (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    input_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)
    filename = os.path.join(save_dir, mask_name)
    img = cv2.cvtColor(input_numpy, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
    '''