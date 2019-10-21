import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset_CAGAN
import utils

def Pre_train(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    colorizationnet = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        colorizationnet = nn.DataParallel(colorizationnet)
        colorizationnet = colorizationnet.cuda()
    else:
        colorizationnet = colorizationnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(colorizationnet.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, colornet):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(colornet.module, 'Pre_CAGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(colornet.module, 'Pre_CAGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(colornet, 'Pre_CAGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(colornet, 'Pre_CAGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the image list
    imglist = utils.text_readlines("ILSVRC2012_train_sal_name.txt")

    # Define the dataset
    trainset = dataset_CAGAN.LabAndSalDataset(opt.colorbaseroot, opt.salbaseroot, imglist)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_L, true_target) in enumerate(dataloader):
            
            # Unzip true_target
            true_Sal = true_target[0]
            true_Sal2 = torch.cat((true_Sal, true_Sal), 1)
            true_ab = true_target[1]

            # To device
            true_L = true_L.cuda()
            true_Sal = true_Sal.cuda()
            true_Sal2 = true_Sal2.cuda()
            true_ab = true_ab.cuda()

            # Add random Gaussian noise
            noise = Tensor(np.random.normal(0, 0.1, (true_L.size(0), 1, 32, 32)))
            noise = noise.cuda()
     
            # Train Generator
            optimizer_G.zero_grad()
            col, sal = colorizationnet(true_L, noise)
            
            # Attention part
            true_Attn_ab = true_ab.mul(true_Sal2)
            sal2 = torch.cat((sal, sal), 1)
            fake_Attn_ab = col.mul(sal2)

            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(col, true_ab)
            Attention_Loss = criterion_L1(fake_Attn_ab, true_Attn_ab)
            Attention_Loss = Pixellevel_L1_Loss + Attention_Loss

            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss + opt.lambda_attn * Attention_Loss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [Attention Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader),
                Pixellevel_L1_Loss.item(), Attention_Loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), colorizationnet)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)

def Continue_train_LSGAN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()
    criterion_CE = torch.nn.CrossEntropyLoss().cuda()

    # Initialize Generator, Discriminator, and Perceptualnet (for perceptual loss)
    colorizationnet = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)

    # To device
    if opt.multi_gpu:
        colorizationnet = nn.DataParallel(colorizationnet)
        colorizationnet = colorizationnet.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
    else:
        colorizationnet = colorizationnet.cuda()
        discriminator = discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(colorizationnet.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, colornet):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(colornet.module, 'LSGAN_CAGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(colornet.module, 'LSGAN_CAGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(colornet, 'LSGAN_CAGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(colornet, 'LSGAN_CAGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the image list
    imglist = utils.text_readlines("ILSVRC2012_train_sal_name.txt")
    stringlist = utils.text_readlines("mapping_string.txt")
    scalarlist = utils.text_readlines("mapping_scalar.txt")

    # Define the dataset
    trainset = dataset_CAGAN.Lab_Sal_LabelDataset(opt.colorbaseroot, opt.salbaseroot, imglist, stringlist, scalarlist)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_L, true_target) in enumerate(dataloader):
            
            # Unzip true_target, and get true_AttnRGB for Attn Loss
            true_Sal = true_target[0]
            true_Sal2 = torch.cat((true_Sal, true_Sal), 1)
            true_ab = true_target[1]
            true_category = true_target[2]

            # To device
            true_L = true_L.cuda()
            true_Sal = true_Sal.cuda()
            true_Sal2 = true_Sal2.cuda()
            true_ab = true_ab.cuda()

            # Add random Gaussian noise
            noise = Tensor(np.random.normal(0, 0.05, (true_L.size(0), 1, 32, 32)))
            noise = noise.cuda()

            # Adversarial ground truth
            valid = Tensor(np.ones((true_L.size(0), 1, 30, 30)))
            fake = Tensor(np.zeros((true_L.size(0), 1, 30, 30)))

            # Train Discriminator
            for j in range(opt.additional_training_d):
                optimizer_D.zero_grad()

                # Generator output
                col, sal = colorizationnet(true_L, noise)

                # Fake colorizations
                fake_conv1, fake_conv2, fake_conv3, fake_conv4, fake_gan_feature, fake_category_feature, fake_recon_z = discriminator(true_L, col.detach())
                loss_fake = criterion_MSE(fake_gan_feature, fake)
                # True colorizations
                true_conv1, true_conv2, true_conv3, true_conv4, true_gan_feature, true_category_feature, true_recon_z = discriminator(true_L, true_ab)
                loss_true = criterion_MSE(true_gan_feature, valid)
                # Overall Loss and optimize
                loss_D = 0.5 * (loss_fake + loss_true)
                loss_D.backward()
                optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            
            # Generator output
            col, sal = colorizationnet(true_L, noise)
                
            # Attention part
            true_Attn_ab = true_ab.mul(true_Sal2)
            sal2 = torch.cat((sal, sal), 1)
            fake_Attn_ab = col.mul(sal2)
            
            # Pixel-level L1 Loss
            Pixellevel_L1_Loss = criterion_L1(col, true_ab)

            # Attention Loss
            Attention_Loss = criterion_L1(fake_Attn_ab, true_Attn_ab)

            # GAN Loss / Feature Matching Loss / CrossEntropy Loss
            fake_conv1, fake_conv2, fake_conv3, fake_conv4, fake_gan_feature, fake_category_feature, fake_recon_z = discriminator(true_L, col)
            true_conv1, true_conv2, true_conv3, true_conv4, true_gan_feature, true_category_feature, true_recon_z = discriminator(true_L, true_ab)
            GAN_Loss = criterion_MSE(fake_gan_feature, valid)
            Feature_Matching_Loss = criterion_L1(fake_conv1, true_conv1) + criterion_L1(fake_conv2, true_conv2) + criterion_L1(fake_conv3, true_conv3) + criterion_L1(fake_conv4, true_conv4)
            CrossEntropy_Loss = criterion_CE(fake_category_feature, true_category)
            InfoGAN_Loss = criterion_L1(fake_recon_z, noise)

            # Overall Loss and optimize
            loss_G = Pixellevel_L1_Loss + opt.lambda_attn * Attention_Loss + opt.lambda_gan * GAN_Loss + opt.lambda_fm * Feature_Matching_Loss + opt.lambda_ce * CrossEntropy_Loss + opt.lambda_info * InfoGAN_Loss
            loss_G.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Loss D: %.4f] [Loss G: %.4f] [Pixel-level Loss: %.4f] [Attention Loss: %.4f]" %
                ((epoch + 1), opt.epochs, i, len(dataloader), loss_D.item(), GAN_Loss.item(), Pixellevel_L1_Loss.item(), Attention_Loss.item()))
            print("\r[Feature Matching Loss: %.4f] [Cross Entropy Loss: %.4f] [InfoGAN Loss: %.4f] Time_left: %s" %
                (Feature_Matching_Loss.item(), CrossEntropy_Loss.item(), InfoGAN_Loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), colorizationnet)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)

def Continue_train_WGAN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_CE = torch.nn.CrossEntropyLoss().cuda()

    # Initialize Generator, Discriminator, and Perceptualnet (for perceptual loss)
    colorizationnet = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)

    # To device
    if opt.multi_gpu:
        colorizationnet = nn.DataParallel(colorizationnet)
        colorizationnet = colorizationnet.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
    else:
        colorizationnet = colorizationnet.cuda()
        discriminator = discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(colorizationnet.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, colornet):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(colornet.module, 'WGAN_CAGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(colornet.module, 'WGAN_CAGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(colornet, 'WGAN_CAGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(colornet, 'WGAN_CAGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the image list
    imglist = utils.text_readlines("ILSVRC2012_train_sal_name.txt")
    stringlist = utils.text_readlines("mapping_string.txt")
    scalarlist = utils.text_readlines("mapping_scalar.txt")

    # Define the dataset
    trainset = dataset_CAGAN.Lab_Sal_LabelDataset(opt.colorbaseroot, opt.salbaseroot, imglist, stringlist, scalarlist)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_L, true_target) in enumerate(dataloader):
            
            # Unzip true_target, and get true_AttnRGB for Attn Loss
            true_Sal = true_target[0]
            true_Sal2 = torch.cat((true_Sal, true_Sal), 1)
            true_ab = true_target[1]
            true_category = true_target[2]

            # To device
            true_L = true_L.cuda()
            true_Sal = true_Sal.cuda()
            true_Sal2 = true_Sal2.cuda()
            true_ab = true_ab.cuda()

            # Add random Gaussian noise
            noise = Tensor(np.random.normal(0, 0.05, (true_L.size(0), 1, 32, 32)))
            noise = noise.cuda()

            # Train Discriminator
            for j in range(opt.additional_training_d):
                optimizer_D.zero_grad()

                # Generator output
                col, sal = colorizationnet(true_L, noise)

                # Fake colorizations
                fake_conv1, fake_conv2, fake_conv3, fake_conv4, fake_gan_feature, fake_category_feature, fake_recon_z = discriminator(true_L, col.detach())
                # True colorizations
                true_conv1, true_conv2, true_conv3, true_conv4, true_gan_feature, true_category_feature, true_recon_z = discriminator(true_L, true_ab)
                # Overall Loss and optimize
                loss_D = - torch.mean(true_gan_feature) + torch.mean(fake_gan_feature)
                loss_D.backward()
                optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            
            # Generator output
            col, sal = colorizationnet(true_L, noise)
                
            # Attention part
            true_Attn_ab = true_ab.mul(true_Sal2)
            sal2 = torch.cat((sal, sal), 1)
            fake_Attn_ab = col.mul(sal2)
            
            # Pixel-level L1 Loss
            Pixellevel_L1_Loss = criterion_L1(col, true_ab)

            # Attention Loss
            Attention_Loss = criterion_L1(fake_Attn_ab, true_Attn_ab)

            # GAN Loss / Feature Matching Loss / CrossEntropy Loss
            fake_conv1, fake_conv2, fake_conv3, fake_conv4, fake_gan_feature, fake_category_feature, fake_recon_z = discriminator(true_L, col)
            true_conv1, true_conv2, true_conv3, true_conv4, true_gan_feature, true_category_feature, true_recon_z = discriminator(true_L, true_ab)
            GAN_Loss = - torch.mean(fake_gan_feature)
            Feature_Matching_Loss = criterion_L1(fake_conv1, true_conv1) + criterion_L1(fake_conv2, true_conv2) + criterion_L1(fake_conv3, true_conv3) + criterion_L1(fake_conv4, true_conv4)
            CrossEntropy_Loss = criterion_CE(fake_category_feature, true_category)
            InfoGAN_Loss = criterion_L1(fake_recon_z, noise)

            # Overall Loss and optimize
            loss_G = Pixellevel_L1_Loss + opt.lambda_attn * Attention_Loss + opt.lambda_gan * GAN_Loss + opt.lambda_fm * Feature_Matching_Loss + opt.lambda_ce * CrossEntropy_Loss + opt.lambda_info * InfoGAN_Loss
            loss_G.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Loss D: %.4f] [Loss G: %.4f] [Pixel-level Loss: %.4f] [Attention Loss: %.4f]" %
                ((epoch + 1), opt.epochs, i, len(dataloader), loss_D.item(), GAN_Loss.item(), Pixellevel_L1_Loss.item(), Attention_Loss.item()))
            print("\r[Feature Matching Loss: %.4f] [Cross Entropy Loss: %.4f] [InfoGAN Loss: %.4f] Time_left: %s" %
                (Feature_Matching_Loss.item(), CrossEntropy_Loss.item(), InfoGAN_Loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), colorizationnet)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)
