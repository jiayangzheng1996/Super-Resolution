import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from model.loss import EGeneratorLoss
from model.model import Generator, Discriminator
# from baseline_model import Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=8, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--discriminator_cycle', default=1, type=int, help='training schedule for discriminator')
parser.add_argument('--generator_cycle', default=1, type=int, help='training schedule for generator')


if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    decay_interval = NUM_EPOCHS // 8
    decay_indices = [decay_interval, decay_interval*2, decay_interval*4, decay_interval*6]

    discriminator_cycle = opt.discriminator_cycle
    generator_cycle = opt.generator_cycle

    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator(num_rrdb_blocks=16, scaling_factor=UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator(opt.crop_size)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = EGeneratorLoss()
    discriminator_criterion = torch.nn.BCEWithLogitsLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
        discriminator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters(), lr=0.0001)
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001)
    optimizerG_decay = optim.lr_scheduler.MultiStepLR(optimizerG, decay_indices, gamma=0.5)
    optimizerD_decay = optim.lr_scheduler.MultiStepLR(optimizerD, decay_indices, gamma=0.5)

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = \
            {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        it = 0
        for data, target in train_bar:
            it += 1
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()

            real_label = torch.full((batch_size, 1), 1, dtype=z.dtype).cuda()
            fake_label = torch.full((batch_size, 1), 0, dtype=z.dtype).cuda()

            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img)
            fake_out = netD(fake_img)
            d_loss_real = discriminator_criterion(real_out-torch.mean(fake_out), real_label)
            d_loss_fake = discriminator_criterion(fake_out-torch.mean(real_out), fake_label)
            d_loss = (d_loss_real + d_loss_fake)/2
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_img, real_img, fake_out, real_out, real_label)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = torch.sigmoid(netD(fake_img)).mean()


            optimizerG.step()


            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += torch.sigmoid(real_out).mean().item() * batch_size
            running_results['g_score'] += fake_out.mean().item() * batch_size

            train_bar.set_description(desc=
                                      '[%d/%d] Loss_D: %.4f Loss_G: %.4f '
                                      'D(x): %.4f D(G(z)): %.4f '
                                      % (
                epoch, NUM_EPOCHS,
                running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']
                                      ))

            batches_done = (epoch-1) * 100 + it

            if batches_done % 50 == 0:
                imgs_lr = torch.nn.functional.interpolate(z, scale_factor=8)
                gen_hr = utils.make_grid(fake_img, nrow=1, normalize=True)
                imgs_lr = utils.make_grid(imgs_lr, nrow=1, normalize=True)
                real_hr = utils.make_grid(real_img, nrow=1, normalize=True)
                img_grid = torch.cat((imgs_lr, gen_hr, real_hr), -1)
                utils.save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

        optimizerG_decay.step()
        optimizerD_decay.step()

        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = \
                {'mse': 0,
                 'ssims': 0,
                 'psnr': 0,
                 'ssim': 0,
                 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = \
                    10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        # save model parameters
        #torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        #torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
