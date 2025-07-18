import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import transforms
from .networks_.dexined import DexiNed, init_dexined
import lpips

class DLP_GAN(BaseModel):
    def name(self):
        return 'Domain Style Transfer Network.'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # Code (paper): G_A (G), G_B (F), Vgg, D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, 'DLP_GAN_G_A', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, 'DLP_GAN_G_B', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        # Load DexiNed
        self.dexinedNet = DexiNed()
        init_dexined(opt.model_dir)
        print(opt.model_dir)
        self.dexinedNet.load_state_dict(torch.load(os.path.join(opt.model_dir, "dexined.weight")))
        self.dexinedNet.cuda()
        # Freeze DexiNed parameters
        for param in self.dexinedNet.parameters():
            param.requires_grad = False

        # Initialize LPIPS loss
        self.lpips_loss = lpips.LPIPS(net='vgg')
        self.lpips_loss.cuda()
        
        # Load VGG16
        self.vggNet = self.lpips_loss.net
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionContent = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        networks.print_network(self.vggNet)
        networks.print_network(self.dexinedNet)

        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_B(fake_B).data
        self.fake_B = fake_B.data

        real_B = Variable(self.input_B, volatile=True)
        fake_A = self.netG_B(real_B)
        self.rec_B = self.netG_A(fake_A).data
        self.fake_A = fake_A.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.data

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.data

    def backward_G(self):
        opt = self.opt
        # Identity loss
        if opt.beta > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B)
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A)

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.data
            self.loss_idt_B = loss_idt_B.data
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)


        # Forward Feature loss
        rec_A = self.netG_B(fake_B)
        real_A_feature = Variable(self.vggNet.forward(transforms.trans_vgg(self.real_A))[2].data, requires_grad=False)
        rec_A_feature = Variable(self.vggNet.forward(transforms.trans_vgg(rec_A))[2].data, requires_grad=True)
        loss_feature_A = self.criterionCycle(real_A_feature, rec_A_feature)

        # Backward Feature loss
        rec_B = self.netG_A(fake_A)
        real_B_feature = Variable(self.vggNet.forward(transforms.trans_vgg(self.real_B))[2].data, requires_grad=False)
        rec_B_feature = Variable(self.vggNet.forward(transforms.trans_vgg(rec_B))[2].data, requires_grad=True)
        loss_feature_B = self.criterionCycle(real_B_feature, rec_B_feature)


        #content realA_fakeB using DexiNed + LPIPS
        loss_semantic_A = self.dexined_lpips_loss(self.real_A, fake_B)
        #content realB_fakeA using DexiNed + LPIPS  
        loss_semantic_B = self.dexined_lpips_loss(self.real_B, fake_A)

        
        # DLP_GAN paper loss function
        # loss_G = opt.lambda_GAN * (loss_G_A + loss_G_B) \
        #          + opt.lambda_Dual * ((loss_feature_A + loss_feature_B) + (loss_semantic_A + loss_semantic_B)) \
        #          + opt.lambda_id * (loss_idt_A + loss_idt_B)
        
        # DSTN paper loss function
        loss_G = loss_G_A + loss_G_B \
                 + opt.alpha_G * loss_feature_A + opt.alpha_F * loss_feature_B\
                 + opt.beta * (loss_idt_A + loss_idt_B) \
                 + opt.gamma * (loss_semantic_A + loss_semantic_B) # Eq (11) in the paper
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data
        self.loss_G_A = loss_G_A.data
        self.loss_G_B = loss_G_B.data
        self.loss_cycle_A = loss_feature_A.data
        self.loss_cycle_B = loss_feature_B.data
        self.loss_Content_A = loss_semantic_A.data
        self.loss_Content_B = loss_semantic_B.data

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B),
                                  ('Content_A', self.loss_Content_A), ('Content_B', self.loss_Content_B)])
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def dexined_lpips_loss(self, real_img, fake_img):
        """
        Compute LPIPS loss using DexiNed output
        
        Args:
            real_img: Real image tensor
            fake_img: Fake image tensor
            
        Returns:
            LPIPS loss value
        """
        # Pass through DexiNed and get the last output
        real_dexined_output = self.dexinedNet(transforms.trans_dexinet(real_img))[-1]
        fake_dexined_output = self.dexinedNet(transforms.trans_dexinet(fake_img))[-1]
        
        # Convert single channel edge maps to 3-channel for LPIPS
        # Repeat the channel dimension to make it RGB-like
        real_dexined_3ch = real_dexined_output.repeat(1, 3, 1, 1)
        fake_dexined_3ch = fake_dexined_output.repeat(1, 3, 1, 1)
        
        # Compute LPIPS loss directly on DexiNed outputs
        # LPIPS internally extracts VGG features and computes loss for all layers
        lpips_loss = self.lpips_loss(real_dexined_3ch, fake_dexined_3ch)
        
        return lpips_loss.mean()
