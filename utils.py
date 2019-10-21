import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

import network_CAGAN

# There are many functions:
# ----------------------------------------
# 1. text_readlines:
# In: a str nominating the a txt
# Parameters: None
# Out: list
# ----------------------------------------
# 2. create_generator:
# In: opt, init_type, init_gain
# Parameters: init type and gain, we highly recommend that Gaussian init with standard deviation of 0.02
# Out: colorizationnet
# ----------------------------------------
# 3. create_discriminator:
# In: opt, init_type, init_gain
# Parameters: init type and gain, we highly recommend that Gaussian init with standard deviation of 0.02
# Out: discriminator_coarse_color, discriminator_coarse_sal, discriminator_fine_color, discriminator_fine_sal
# ----------------------------------------
# 4. create_perceptualnet:
# In: None
# Parameters: None
# Out: perceptualnet
# ----------------------------------------
# 5. load_dict
# In: process_net (the net needs update), pretrained_net (the net has pre-trained dict)
# Out: process_net (updated)
# ----------------------------------------
# 6. savetxt
# In: list
# Out: txt
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

def create_generator(opt):
    if opt.pre_train:
        # Initialize the networks
        colorizationnet = network_CAGAN.GeneratorAttnUNet()
        # Init the networks
        network_CAGAN.weights_init(colorizationnet, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        #colorizationnet = torch.load('Pre_CAGAN_' + opt.load_name + '.pth')
        colorizationnet = network_CAGAN.GeneratorAttnUNet()
        print('Generator is loaded!')
    return colorizationnet

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network_CAGAN.LabDiscriminator()
    # Load the VGG prior to the encoders
    pretrained_net = torch.load(opt.load_LabVGG_name + '.pth')
    load_dict(discriminator, pretrained_net)
    # It does not gradient
    for param in discriminator.parameters():
        param.requires_grad = False
    return discriminator

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)
