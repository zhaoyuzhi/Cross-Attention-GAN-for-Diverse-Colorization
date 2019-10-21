import os
from PIL import Image

# read a txt expect EOF
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

# resize all images
def resize_all_images(basepath, savepath, imglist, resize_amount):
    for i in range(len(imglist)):
        imgpath = basepath + imglist[i]
        finalpath = savepath + imglist[i]
        img = Image.open(imgpath[i])
        channel = len(img.split())
        if channel == 1:
            img = img.convert('L')
        if channel > 1:
            img = img.convert('RGB')
        if i % 10000 == 0:
            print('%dth image has been processed.' % i)
        img = img.resize((resize_amount, resize_amount), Image.ANTIALIAS)
        img.save(finalpath)

if __name__ == '__main__':
    # Read the raw images from txt
    imglist = text_readlines("./COCO2014_val_name.txt")
    print('the number of all image:', len(imglist))
    
    # resize all images
    basepath = 'D:\\datasets\\COCO2014_val_256\\'
    savepath = 'D:\\datasets\\COCO2014_val_256\\'
    resize_amount = 256
    resize_all_images(basepath, savepath, imglist, resize_amount)
