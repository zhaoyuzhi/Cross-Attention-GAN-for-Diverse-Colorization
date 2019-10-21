import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from skimage import color

import network_CAGAN

# ----------------------------------------
#                 Testing
# ----------------------------------------

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def test(gray_img, colornet):
    # out_ab_: [1, 2, 256, 256]; sal_: [1, 1, 256, 256]
    col, sal = colornet(gray_img)
    out_ab = col.cpu().detach().numpy().reshape([2, 256, 256])
    sal = sal.cpu().detach().numpy().reshape([256, 256])
    out_ab = out_ab.transpose(1, 2, 0)
    out_ab = out_ab * 110
    # gray_img: [1, 1, 256, 256]
    gray_img = gray_img.cpu().numpy().reshape([256, 256, 1])
    gray_img = (gray_img + 1) * 50
    # concatenate
    out_lab = np.concatenate((gray_img, out_ab), axis = 2)
    out_lab = out_lab.astype(np.float64)
    out_rgb = color.lab2rgb(out_lab) * 255
    out_rgb = out_rgb.astype(np.uint8)
    sal = sal * 255
    sal = np.array(sal, dtype = np.uint8)
    return out_rgb, sal
    
def getImage(root):
    transform = transforms.ToTensor()

    gray_img = Image.open(root)
    rgb = gray_img.resize((256, 256), Image.ANTIALIAS).convert('RGB')
    rgb = np.array(rgb)
    lab = color.rgb2lab(rgb).astype(np.float32)
    lab = transform(lab)
    l = lab[[0], ...] / 50 - 1.0
    l = l.reshape([1, 1, 256, 256]).cuda()
    return l

def comparison(root, colornet):
    # Read raw image
    img = Image.open(root).convert('RGB')
    img = img.resize((256, 256), Image.ANTIALIAS)
    # Forward propagation
    torchimg = getImage(root)
    out_rgb, sal = test(torchimg, colornet)
    # Show
    out_rgb = np.concatenate((out_rgb, img), axis = 1)
    img_rgb = Image.fromarray(out_rgb)
    img_rgb.show()
    return img_rgb

def colorization(root, colornet):
    # Read raw image
    img = Image.open(root).convert('RGB')
    width = img.size[0]
    height = img.size[1]
    # Forward propagation
    torchimg = getImage(root)
    out_rgb, sal = test(torchimg, colornet)
    # Show
    img_rgb = Image.fromarray(out_rgb)
    img_rgb = img_rgb.resize((width, height), Image.ANTIALIAS)
    img_rgb.show()
    return img_rgb

def generation(baseroot, saveroot, imglist, colornet):
    for i in range(len(imglist)):
		# Read raw image
        readname = baseroot + imglist[i]
        print(readname)
        img = Image.open(readname).convert('RGB')
        width = img.size[0]
        height = img.size[1]
        # Forward propagation
        torchimg = getImage(readname)
        out_rgb, sal = test(torchimg, colornet)
        # Save
        img_rgb = Image.fromarray(out_rgb)
        img_rgb = img_rgb.resize((width, height), Image.ANTIALIAS)
        savename = saveroot + imglist[i]
        img_rgb.save(savename)
    print('Done!')

if __name__ == "__main__":

    # Define the basic variables
    root = 'C:\\Users\\yzzha\\Desktop\\dataset\\COCO2014_val_256\\COCO_val2014_000000000257.jpg'
    root = 'C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256\\ILSVRC2012_val_00002187.JPEG'
    #root = 'C:\\Users\\yzzha\\Desktop\\dataset\\colorization_test\\Landscape\\2.jpg'
    colornet = torch.load('Pre_CAGAN_epoch3_bs8.pth')
    colornet.eval()
    
    '''
    # Define generation variables
    txtname = 'ILSVRC2012_val_name.txt'
    imglist = text_readlines(txtname)
    baseroot = 'D:\\datasets\\ILSVRC2012\\ILSVRC2012_val_256\\'
    saveroot = 'D:\\datasets\\ILSVRC2012\\ILSVRC2012_val_256_colorization\\'
    '''

    # Choose a task:
    choice = 'colorization'
    save = False

    # comparison: Compare the colorization output and ground truth
    # colorization: Show the colorization as original size
    # generation: Generate colorization results given a folder
    if choice == 'comparison':
        img_rgb = comparison(root, colornet)
        if save:
            imgname = root.split('/')[-1]
            img_rgb.save('./' + imgname)
    if choice == 'colorization':
        img_rgb = colorization(root, colornet)
        if save:
            imgname = root.split('/')[-1]
            img_rgb.save('./' + imgname)
    if choice == 'generation':
        generation(baseroot, saveroot, imglist, colornet)
    