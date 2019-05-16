import argparse
import itertools
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models import *
from datasets import *
from utils import *
import torch
import numpy as np
import time
import datetime
import sys
import ipdb

def train(opt):
    # Initialize generator and discriminator
    input_shape = (3, opt.img_height, opt.img_width)
    G_AB = GeneratorResNet(opt.n_residual_blocks)
    G_BA = GeneratorResNet(opt.n_residual_blocks)
    D_A, D_B = Discriminator(input_shape), Discriminator(input_shape)
    # Initialize weights
    G_AB.apply(weights_init_normal), G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal), D_B.apply(weights_init_normal)
    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
        G_AB, G_BA = G_AB.cuda(), G_BA.cuda()
        D_A, D_B = D_A.cuda(), D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    # Image transformations
    transforms_ = transforms.Compose([
        transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((opt.img_height, opt.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Training data loader
    dataloader = DataLoader(
        ImageDataset("./data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
        batch_size=opt.batch_size, shuffle=True, num_workers=8, )
    # Test data loader
    val_dataloader = DataLoader(
        ImageDataset("./data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
        batch_size=5, shuffle=True, num_workers=1, )
    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.epochs, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.epochs, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.epochs, opt.decay_epoch).step)

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    #  Training
    prev_time = time.time()
    for epoch in range(opt.epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            # ------------------
            #  Train Generators
            # ------------------
            G_AB.train(), G_BA.train()
            optimizer_G.zero_grad()
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            # Total loss
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity
            loss_G.backward()
            optimizer_G.step()
            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()
            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            loss_D = (loss_D_A + loss_D_B) / 2
            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (epoch, opt.epochs, i, len(dataloader), loss_D.item(), loss_G.item(), loss_GAN.item(), loss_cycle.item(),
                   loss_identity.item(), time_left,))
            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                # sample_images(batches_done)
                imgs = next(iter(val_dataloader))
                G_AB.eval(),G_BA.eval()
                real_A, real_B = Variable(imgs["A"].type(Tensor)), Variable(imgs["B"].type(Tensor))
                fake_A, fake_B = G_BA(real_B), G_AB(real_A)
                # Arange images along x-axis
                real_A = make_grid(real_A, nrow=5, normalize=True)
                real_B = make_grid(real_B, nrow=5, normalize=True)
                fake_A = make_grid(fake_A, nrow=5, normalize=True)
                fake_B = make_grid(fake_B, nrow=5, normalize=True)
                # Arange images along y-axis
                image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
                save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)
        # Update learning rates
        lr_scheduler_G.step(), lr_scheduler_D_A.step(), lr_scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="orange2apple", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--img_width", type=int, default=128, help="size of image width")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    opt = parser.parse_args()
    # Create sample and checkpoint directories
    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)
    train(opt)

if __name__ == "__main__":
    main()
