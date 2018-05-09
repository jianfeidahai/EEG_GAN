import utils, torch, time, os, pickle, imageio, math
from scipy.misc import imsave
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pdb
from utils import Flatten
import matplotlib.pyplot as plt

#BatchNorm -> LayerNorm or pixelnorm

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_dim = 100
        self.input_height = 1
        self.input_width = 1
        self.output_dim = 3


        # Upsample + conv2d is better than convtranspose2d
        self.deconv = nn.Sequential(
            # 4
            nn.Conv2d(self.input_dim, 512, 4, 1, 3, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            #nn.BatchNorm2d(512),
            nn.ReLU(),

            # 8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            #nn.BatchNorm2d(256),
            nn.ReLU(),

            # 16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            #nn.BatchNorm2d(128),
            nn.ReLU(),

            # 32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, self.output_dim, 3, 1, 1, bias=False),
            nn.Sigmoid(),
        )


    def forward(self, feature):
        feature = feature.view(-1, self.input_dim, 1, 1)
        x = self.deconv(feature)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_cls):
        super(Discriminator, self).__init__()
        self.input_dim = 3
        self.num_cls = num_cls

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, 4, 2, 1, bias=False), # 64 -> 32
            nn.InstanceNorm2d(32, affine=True),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 32 -> 16
            nn.InstanceNorm2d(64, affine=True),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 16 -> 8
            nn.InstanceNorm2d(128, affine=True),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 8 -> 4
            nn.InstanceNorm2d(256, affine=True),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.convCls = nn.Sequential(
            nn.Conv2d(256, self.num_cls, 4, bias=False),
            #Flatten()
            #nn.Linear(256, self.num_cls),
            #nn.Softmax2d()
        )

        self.convGAN = nn.Sequential(
            nn.Conv2d(256, 1, 4, bias=False),
            nn.Sigmoid(),
            #Flatten()
        )

    def forward(self, y_):
        feature = self.conv(y_)

        fGAN = self.convGAN(feature).squeeze(3).squeeze(2)
        #fcls = self.convCls(feature).squeeze(3).squeeze(2)

        return fGAN

class GAN(object):
    def __init__(self):#, args):
        #parameters
        self.batch_size = 128 #args.batch_size
        self.epoch = 1000#args.epoch
        
        self.save_dir = './models'#args.save_dir
        self.result_dir = './results'#args.result_dir
        self.dataset = "ImageNet"#args.dataset
        ''' 
        self.dataroot_dir = args.dataroot_dir
        self.log_dir = args.log_dir
        self.multi_gpu = args.multi_gpu
        '''
        self.model_name = "GAN"#args.gan_type
        self.sample_num = 128
        self.gpu_mode = True#args.gpu_mode
        self.num_workers = 0#args.num_workers
        self.beta1 = 0.5 #args.beta1
        self.beta2 = 0.999 #args.beta2
        self.lrG = 0.0002#args.lrG
        self.lrD = 0.00001#0.0002 is good at single sample but many class is better 0.00005 #args.lrD
        self.type = "train"
        self.lambda_ = 0.25
        self.n_critic = 5

        #load dataset
        self.data_loader = DataLoader(utils.ImageNet(transform2=transforms.Compose([transforms.Scale(64), transforms.RandomCrop(64),  transforms.ToTensor()]),
            transform1=transforms.Compose([transforms.ToTensor()]), type_=self.type),
                                      batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.enc_dim = 100 # dimension of output from Encoder
        self.num_cls = self.data_loader.dataset.num_cls # number of class ImageNet

        #networks init

        self.G = Generator()
        self.D = Discriminator(num_cls=self.num_cls)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))

        if self.gpu_mode:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.MSE_loss = nn.MSELoss().cuda()
            self.L1_loss = nn.L1Loss().cuda()
            self.ML_loss = nn.MultiLabelMarginLoss().cuda()
        else:
            self.CE_loss = nn.CrossEntropyLoss()
            self.BCE_loss = nn.BCELoss()
            self.MSE_loss = nn.MSELoss()
            self.L1_loss = nn.L1Loss()
            self.ML_loss = nn.MultiLabelMarginLoss()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        #self.train_hist['E_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))


        #train
        self.D.train()
        start_time = time.time()
        for epoch in range(self.epoch):
            #self.G.train()
            self.G.train()
            epoch_start_time = time.time()
            for iB, (x_, y_, labels, one_hot_vector, class_label) in enumerate(self.data_loader):
                #print()
                #print(x_.shape, y_.shape)

                if iB == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                #Make Laten Space
                #z_ = torch.rand(self.batch_size, self.enc_dim)
                z_ = torch.FloatTensor(self.batch_size, self.enc_dim).normal_(0.0, 1.0)


                if self.gpu_mode:
                    x_, z_, y_, one_hot_vector_, class_label_ = Variable(x_.cuda()), Variable(z_.cuda()), Variable(y_.cuda()), Variable(one_hot_vector.cuda()), Variable(class_label.cuda())
                else:
                    x_, z_, y_, one_hot_vector_, class_label_ = Variable(x_), Variable(z_), Variable(y_), Variable(one_hot_vector), Variable(class_label)


                #Update D_network
                
                self.D_optimizer.zero_grad()
                D_real = self.D(y_)
                #pdb.set_trace()
                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                #C_real_loss = self.ML_loss(C_real, one_hot_vector_) # divided by self.num_cls
                #C_real_loss = self.CE_loss(C_real, class_label_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                #C_fake_loss = self.ML_loss(C_fake, one_hot_vector_)
                #C_fake_loss = self.CE_loss(C_fake, class_label_)



                # gradient penalty
                if self.gpu_mode:
                    alpha = torch.rand(y_.size()).cuda()
                else:
                    alpha = torch.rand(y_.size())

                y_hat = Variable(alpha * y_.data + (1 - alpha) * G_.data, requires_grad=True)

                pred_hat = self.D(y_hat)
                if self.gpu_mode:
                    gradients = grad(outputs=pred_hat, inputs=y_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                else:
                    gradients = grad(outputs=pred_hat, inputs=y_hat, grad_outputs=torch.ones(pred_hat.size()),
                                     create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty = self.lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()




                D_loss = D_real_loss + D_fake_loss + gradient_penalty
                self.train_hist['D_loss'].append(D_loss.data[0])
                
                num_correct_real = torch.sum(D_real > 0.5)
                num_correct_fake = torch.sum(D_fake < 0.5)

                D_acc = float(num_correct_real.data[0] + num_correct_fake.data[0]) / (self.batch_size * 2)


                D_loss.backward()
                if D_acc<0.8:
                    #print("D train!")
                    self.D_optimizer.step()
               

                #Update G Network
                for iG in range(4):
                    self.G_optimizer.zero_grad()
                
                    G_ = self.G(z_)
                    D_fake= self.D(G_)

                    GAN_loss = self.BCE_loss(D_fake, self.y_real_)
                    #C_fake_loss = self.ML_loss(C_fake, one_hot_vector_)
                    #C_fake_loss = self.CE_loss(C_fake, class_label_)
                    #G_recon_loss = self.MSE_loss(G_, y_)
                    #G_recon_loss = self.L1_loss(G_, y_)

                    num_wrong_fake = torch.sum(D_fake > 0.5)
                    G_acc = float(num_wrong_fake.data[0]) / self.batch_size

                    G_loss = GAN_loss# + (C_fake_loss) + G_recon_loss*80
                    if iG == 0:
                        print("[E%03d]"%epoch,"G_loss : ", GAN_loss.data[0], "  D_loss : ", D_loss.data[0], "   D_acc : ", D_acc, "  G_acc : ", G_acc)
                        self.train_hist['G_loss'].append(G_loss.data[0])
                
                    G_loss.backward()
                    self.G_optimizer.step()
                    
                    


                if iB%100 == 0 and epoch>100 and epoch%50 == 0:
                    self.G.eval()
                    #self.G_origin.eval()

                    tot_num_samples = min(self.sample_num, self.batch_size)
                    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
                    
                    save_path = './result_gan/'
	
                    sample_G = self.G(z_)#self.G_origin(x_, z_)s
                    if self.gpu_mode:
                        sample_G = sample_G.cpu().data.numpy().transpose(0,2,3,1)
                        gt = y_.cpu().data.numpy().transpose(0,2,3,1)
                    else:
                        sampel_G = sample_G.data.numpy().transpose(0,2,3,1)
                        gt = y_.data.numpy().transpose(0,2,3,1)

                   #We can check with or without Encoder output at here ex) self.G.Dec(z_) vs self.G(x_,z_)
                    utils.save_images(sample_G[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],save_path+ '_E%03d'%epoch + '_I%03d'%iB + '.png')
                    utils.save_images(gt[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],save_path +'GT_E%03d'%epoch + '_I%03d'%iB + '.png')
                    utils.loss_plot(self.train_hist, save_path)
                    self.save()

                    self.G.train()
                    
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))


A=GAN()
A.train()

