from cmath import log
from select import select
from dataset import cifar
import engine
from torch.cuda.amp import autocast, GradScaler
from mosaic_core import registry
import numpy as np
# import config
import torch
import os
import logging
import torch.nn as nn
from utils.utils import DataIter
from torch.utils.data import DataLoader
import torch.optim as optim
import utils.utils as utils
import copy
import random
from tqdm import tqdm
from mosaic_core.pipeline.privacy_analysis import PrivacyCostAnalysis
# from torchvision import transforms



class OneTeacher:
    def __init__(self, student, teacher, generator, discriminator, args, writer):
        self.writer = writer
        self.args = args
        self.val_loader, self.priv_data, self.local_datanum, self.local_cls_datanum = self.gen_dataset(
            args)
        self.local_datanum = torch.FloatTensor(self.local_datanum).cuda()
        self.local_cls_datanum = torch.FloatTensor(
            self.local_cls_datanum).cuda()
        self.netG = generator
        # 对图像标准化，加速图像收敛
        self.normalizer = utils.Normalizer(args.dataset)
        self.netDS = utils.copy_parties(self.args.N_PARTIES, discriminator)
        # teacher/student
        self.netS = student
        self.init_netTS(teacher)

    def gen_dataset(self, args):
        # 从data/fed/torchdata下获取dataset
        num_classes, ori_training_dataset, val_dataset = registry.get_dataset(name=args.dataset,
                                                                              data_root=args.data_root)
        _, train_dataset, _ = registry.get_dataset(
            name=args.unlabeled, data_root=args.data_root)
        # _, ood_dataset, _ = registry.get_dataset(name=args.unlabeled, data_root=args.data_root)
        # see Appendix Sec 2, ood data is also used for training
        # ood_dataset.transforms = ood_dataset.transform = train_dataset.transform # w/o augmentation
        train_dataset.transforms = train_dataset.transform = val_dataset.transform  # w/ augmentation
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers)
        # return train_dataset,ood_dataset,num_classes,ori_training_dataset,val_dataset
        local_datanum = np.zeros(self.args.N_PARTIES)
        local_cls_datanum = np.zeros((self.args.N_PARTIES, self.args.n_classes))
        for localid in range(self.args.N_PARTIES):
            # count
            local_datanum[localid] = 50000
            # class specific count
            for cls in range(self.args.n_classes):
                local_cls_datanum[localid, cls] = 5000

        return val_loader, [train_dataset], local_datanum, local_cls_datanum

    def init_netTS(self, teacher):
        # XXX 修改了初始化教师模型逻辑，
        if self.args.from_teacher_ckpt == '':
            self.netTS = utils.copy_parties(
                self.args.N_PARTIES, teacher)  # can be different
        else:
            self.netTS = []
            ckpts = os.listdir(self.args.from_teacher_ckpt)
            for n in range(self.args.N_PARTIES):
                teacher = registry.get_model(
                    self.args.teacher, num_classes=self.args.n_classes, pretrained=True).eval()
                teacher = teacher.cuda(self.args.gpu)
                cur_teacher = ''
                for ckpt in ckpts:
                    if str(n) == ckpt.split('.')[0]:
                        cur_teacher = os.path.join(
                            self.args.from_teacher_ckpt, ckpt)
                        utils.load_dict(cur_teacher, teacher, self.args.gpu)
                        self.netTS.append(teacher)
                        break
                print(cur_teacher)


    def init_training(self):
        # 初始化训练
        global noise_multiplier
        noise_multiplier = self.args.noise_multiplier
        self.local_dataloader = []
        self.local_ood_dataloader = []
        for n in range(self.args.N_PARTIES):
            tr_dataset = self.priv_data[n]
            local_loader = DataLoader(
                dataset=tr_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.workers,
                sampler=None)
            self.local_dataloader.append(DataIter(local_loader))
        # FIXME! only limited data used
        # TODO min -> max
        self.steps_per_disc = [len(local_loader.dataloader)
                               for local_loader in self.local_dataloader]
        self.cur_step_per_disc = [0 for _ in range(len(self.local_dataloader))]
        if self.args.use_maxIters:
            self.iters_per_round = max(
                [len(local_loader.dataloader) for local_loader in self.local_dataloader])
        else:
            self.iters_per_round = min(
                [len(local_loader.dataloader) for local_loader in self.local_dataloader])

        steps_all = self.args.epochs * self.iters_per_round
        # init optim
        if self.args.optimizer == 'SGD':
            self.optim_s = optim.SGD(
                self.netS.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
        else:
            self.optim_s = optim.Adam(
                self.netS.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

        # TODO 去除学习率衰减
        if not self.args.fixed_lr:
            self.sched_s = optim.lr_scheduler.CosineAnnealingLR(
                self.optim_s, steps_all, eta_min=self.args.lr_min)  # TODO, ,
        # for gen
        if self.args.use_pretrained_generator:
            loc = 'cuda:{}'.format(self.args.gpu)
            checkpoint = torch.load(self.args.use_pretrained_generator,
                                    map_location=loc)
            self.netG.load_state_dict(checkpoint)

        self.optim_g = torch.optim.Adam(
            self.netG.parameters(), lr=self.args.lr_g, betas=[0.5, 0.999])

        # TODO 去除generator学习率衰减
        if not self.args.fixed_lr:
            self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim_g, T_max=steps_all)
        param_ds = []

        for n in range(self.args.N_PARTIES):
            param_ds += list(self.netDS[n].parameters())

        self.optim_d = torch.optim.Adam(
            param_ds, lr=self.args.lr_g, betas=[0.5, 0.999])

        # TODO 去除discriminator学习率衰减
        if not self.args.fixed_lr:
            self.sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim_d, T_max=steps_all)
        # TODO: not all complete steps_all if local_percetnt<1

        # criterion
        self.criterion_distill = nn.L1Loss(reduction='mean')
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.criterion_bce = nn.functional.binary_cross_entropy_with_logits

        # netS training records
        self.bestacc = 0
        self.best_statdict = copy.deepcopy(self.netS.state_dict())
        # save path
        self.savedir = f'{self.args.gen_ckpt_path}/{self.args.logfile}'
        self.savedir_gen = f'{self.args.gen_ckpt_path}/{self.args.logfile}/gen'
        if not os.path.isdir(self.savedir_gen):
            os.makedirs(self.savedir_gen)

    def update(self):
        self.init_training()

        self.global_step = 0
        # default: local_percent =1
        selectN = list(range(0, self.args.N_PARTIES))
        self.localweight = self.local_datanum / \
            self.local_datanum.sum()  # nlocal*nclass
        self.localclsweight = self.local_cls_datanum / \
            self.local_cls_datanum.sum(dim=0)  # nlocal*nclass
        # resume trainning if args.resume exists

        self.resume_trainning()
        # use fp16 to accelerate computations
        if self.args.fp16:
            self.args.scaler_s = GradScaler() if self.args.fp16 else None
            self.args.scaler_g = GradScaler() if self.args.fp16 else None
            self.args.scaler_d = GradScaler() if self.args.fp16 else None
            self.args.autocast = autocast
        else:
            self.args.autocast = engine.utils.dummy_ctx

        for round in range(self.args.start_epoch, self.args.epochs):
            if self.args.local_percent < 1:
                # FIXME 此处由于种子被固定，因此随机出来的selectN是固定的
                selectN = random.sample(selectN, int(
                    self.args.local_percent*self.args.N_PARTIES))

                countN = self.local_datanum[selectN]
                self.localweight = countN/countN.sum()  # nlocal
                countN = self.local_cls_datanum[selectN]
                self.localclsweight = countN/countN.sum(dim=0)  # nlocal*nclass
            logging.info(
                f'************Start Round {round} -->> {self.args.epochs}***************')
            self.update_round(round, selectN)

        # save G,D in the end
        torch.save(self.netG.state_dict(), f'{self.savedir_gen}/generator.pt')
        for n in range(self.args.N_PARTIES):
            torch.save(self.netDS[n].state_dict(),
                       f'{self.savedir_gen}/discrim{n}.pt')
        ############################# exiting ####################################

    def resume_trainning(self):
        ############################################
        # Resume Train from checkpoints
        ############################################
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                if self.args.gpu is None:
                    checkpoint = torch.load(self.args.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(self.args.gpu)
                    checkpoint = torch.load(self.args.resume, map_location=loc)

                try:
                    self.netS.module.load_state_dict(
                        checkpoint['s_state_dict'])
                    self.netG.module.load_state_dict(
                        checkpoint['g_state_dict'])
                    for n in range(self.args.N_PARTIES):
                        self.netDS[n].module.load_state_dict(
                            checkpoint['ds_state_dict'][n])
                except:
                    self.netS.load_state_dict(checkpoint['s_state_dict'])
                    self.netG.load_state_dict(checkpoint['g_state_dict'])
                    for n in range(self.args.N_PARTIES):
                        self.netDS[n].load_state_dict(
                            checkpoint['ds_state_dict'][n])
                best_acc1 = checkpoint['best_acc']
                try:
                    self.args.start_epoch = checkpoint['epoch']
                    self.bestacc = checkpoint['best_acc']

                    self.optim_g.load_state_dict(checkpoint['optim_g'])
                    # self.sched_g.load_state_dict(checkpoint['sched_g'])
                    self.optim_s.load_state_dict(checkpoint['optim_s'])
                    # self.sched_s.load_state_dict(checkpoint['sched_s'])
                    # self.sched_d.load_state_dict(checkpoint['sched_d'])
                    self.optim_d.load_state_dict(checkpoint['optim_d'])
                    # 修改学习率
                    if self.args.modify_optim_lr:
                        # 目前仅修改student的学习率
                        for param_s in self.optim_s.param_groups:
                            print('ori_optim_s LR:', param_s['lr'])
                            param_s['lr'] = self.args.lr
                            print('new_optim_s LR:', param_s['lr'])
                        for param_g in self.optim_g.param_groups:
                            print('ori_optim_g LR:', param_g['lr'])
                            param_g['lr'] = self.args.lr_g
                            print('new_optim_g LR:', param_g['lr'])
                        for param_d in self.optim_d.param_groups:
                            print('ori_optim_d LR:', param_d['lr'])
                            param_d['lr'] = self.args.lr_g
                            print('new_optim_d LR:', param_d['lr'])
                except:
                    print("Fails to load additional model information")
                print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                      .format(self.args.resume, checkpoint['epoch'], best_acc1))
            else:
                print("[!] no checkpoint found at '{}'".format(self.args.resume))

    def change_status_to_train(self):
        # model.train(): 启用batch normalization以及dropout
        # model.eval(): 不启用batch normalization以及dropout，用于测试
        self.netG.train()
        for n in range(self.args.N_PARTIES):
            self.netDS[n].train()
            self.netTS[n].eval()
        self.netS.train()

    def update_round(self, roundd, selectN):
        #
        self.change_status_to_train()
        bestacc_round = self.bestacc
        self.cur_step_per_disc = [0 for _ in range(len(self.local_dataloader))]
        for iter in tqdm(range(self.iters_per_round)):
            # 1. update D,G
            z = torch.randn(size=(self.args.batch_size, self.args.z_dim)).cuda()
            # Generator从随机噪声生成图像,一次生成一个batchsize的图像
            syn_img = self.netG(z)
            # 归一化生成图像
            syn_img = self.normalizer(syn_img)
            # 生成图像送入Discriminator，更新Discriminators
            self.update_netDS_batch(syn_img, selectN)
            # 批量更新Generator
            self.update_netG_batch(syn_img, selectN)
            # 2. Distill, update S
            self.update_netS_batch(selectN)
            if not self.args.fixed_lr:
                self.sched_d.step()
                self.sched_g.step()
                self.sched_s.step()
            # val

        acc = validate(self.optim_s.param_groups[0]['lr'], self.val_loader,
                       self.netS, self.criterion, self.args,roundd)
        
        self.global_step += 1
        if self.args.clip:
            privacy_cost = PrivacyCostAnalysis().gswgan_cost(self.args.batch_size, self.args.N_PARTIES,
                                                            train_iters=self.global_step, noise_multiplier=self.args.noise_multiplier)
        else:
            privacy_cost = None

        is_best = False
        if acc > bestacc_round:
            logging.info(f'Iter{iter}, best for now:{acc}')
            self.best_statdict = copy.deepcopy(self.netS.state_dict())
            bestacc_round = acc
            is_best = True

        # reload G,D?
        # self.netS.load_state_dict(self.best_statdict, strict=True)
        logging.info(f'=============Round{roundd}, BestAcc originate: {(self.bestacc):.2f}, to{(bestacc_round):.2f}, privacy-cost: {privacy_cost}====================')

        if bestacc_round > self.bestacc:
            print(f'selectN:{selectN}')
            savename = os.path.join(
                self.savedir, f'r{roundd}_{(bestacc_round):.2f}.pt')
            torch.save(self.best_statdict, savename)
            self.bestacc = bestacc_round

        if self.writer is not None:
            self.writer.add_scalar('BestACC', self.bestacc, roundd)

        
        checkpoints = {
            "epoch": roundd + 1,
            "arch": self.args.student,
            "s_state_dict": self.best_statdict,
            "g_state_dict": self.netG.state_dict(),
            "ds_state_dict": [self.netDS[i].state_dict() for i in range(self.args.N_PARTIES)],
            "best_acc": float(bestacc_round),
            "optim_s": self.optim_s.state_dict(),
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
        }
        save_checkpoint(checkpoints,
                        is_best,
                        self.args.ckpt_path,
                        filename=f'E{roundd}-Dbs{self.args.batch_size}-Ts{self.args.N_PARTIES}-ACC{round(float(self.bestacc), 2)}.pth')

    def update_netDS_batch(self, syn_img, selectN):
        loss = 0.
        with self.args.autocast():
            cnt_disc = 0
            for localid in selectN:
                if self.cur_step_per_disc[localid] >= self.steps_per_disc[localid]:
                    continue
                cnt_disc += 1
                # 对fake照片有一个输出d_out_fake, shape=[batchsize:16,channel:1,8,8] 8*8 patch输出
                d_out_fake = self.netDS[localid](syn_img.detach())
                real_img = self.local_dataloader[localid].next()[0].cuda()  # list [img, label, ?]
                self.cur_step_per_disc[localid] += 1
                # 对real照片输出d_out_real
                d_out_real = self.netDS[localid](real_img.detach())

                loss_d = (self.criterion_bce(d_out_fake, torch.zeros_like(d_out_fake), reduction='sum') +
                          self.criterion_bce(d_out_real, torch.ones_like(d_out_real), reduction='sum')) / (
                    2 * len(d_out_fake))

                loss += loss_d
            loss *= self.args.w_disc/cnt_disc

        self.optim_d.zero_grad()
        if self.args.fp16:
            scaler_d = self.args.scaler_d
            scaler_d.scale(loss).backward()
            scaler_d.step(self.optim_d)
            scaler_d.update()
        else:
            loss.backward()
            self.optim_d.step()

        if self.writer is not None:
            self.writer.add_scalar('LossDiscriminator',
                                   loss.item(), self.global_step)

    def update_netG_batch(self, syn_img, selectN):
        # 1. gan loss
        loss_gan = []
        with self.args.autocast():
            for localid in selectN:
                d_out_fake = self.netDS[localid](syn_img)  # gradients in syn
                # d_out_fake.shape = [batchsize, 1, 8, 8]
                # 为什么要计算d_out_fake与全1矩阵的bceloss?
                loss_gan.append(self.criterion_bce(d_out_fake, torch.ones_like(d_out_fake),
                                                   reduction='sum') / len(d_out_fake))
            # import pdb; pdb.set_trace()
            if self.args.is_emsember_generator_GAN_loss == "y":
                loss_gan = self.ensemble_locals(torch.stack(loss_gan))
            else:
                loss_gan = torch.sum(torch.stack(loss_gan))

            loss_align = []
            loss_balance = []

            # 2. adversarial distill loss
            if self.args.use_pate:
                logits_S = self.netS(syn_img)
                clean_logits, logits_T = self.pate_teacher_outs(syn_img, self.args.lap_scale, selectN)
                loss_adv = - self.criterion(logits_S, logits_T.detach())

                for n in range(len(selectN)):
                    t_out = clean_logits[n]
                    pyx = torch.nn.functional.softmax(t_out, dim=1)  # p(y|G(z))
                    log_softmax_pyx = torch.nn.functional.log_softmax(t_out, dim=1)
                    py = pyx.mean(0)
                    # To generate distinguishable imgs
                    loss_align.append(-(pyx * log_softmax_pyx).sum(1).mean()) # 信息熵（交叉熵）
                    # Alleviating Mode Collapse for unconditional GAN
                    loss_balance.append((py * torch.log2(py)).sum())

                if self.args.gen_loss_avg:
                    # 使用该参数会使loss平均
                    loss_align = sum(loss_align)/len(loss_align)
                    loss_balance = sum(loss_balance)/len(loss_balance)
                else:
                    loss_align = self.ensemble_locals(torch.stack(loss_align))
                    loss_balance = self.ensemble_locals(torch.stack(loss_balance))

            else:  # use k-L divergence loss kldiv类似软标签的交叉熵函数
                logits_T = self.forward_teacher_outs(syn_img, selectN)
                ensemble_logits_T = self.ensemble_locals(logits_T)
                # 
                logits_S = self.netS(syn_img)
                loss_adv = - \
                    engine.criterions.kldiv(
                        logits_S, ensemble_logits_T, T=self.args.T)
            
                # 3.regularization for each t_out (not ensembled) #TO DISCUSS
                for n in range(len(selectN)):
                    t_out = logits_T[n]
                    pyx = torch.nn.functional.softmax(t_out, dim=1)  # p(y|G(z))
                    log_softmax_pyx = torch.nn.functional.log_softmax(t_out, dim=1)
                    py = pyx.mean(0)
                    # To generate distinguishable imgs
                    loss_align.append(-(pyx * log_softmax_pyx).sum(1).mean()) # 信息熵（交叉熵）
                    # Alleviating Mode Collapse for unconditional GAN
                    loss_balance.append((py * torch.log2(py)).sum())

                if self.args.gen_loss_avg:
                    # 使用该参数会使loss平均
                    loss_align = sum(loss_align)/len(loss_align)
                    loss_balance = sum(loss_balance)/len(loss_balance)
                else:
                    loss_align = self.ensemble_locals(torch.stack(loss_align))
                    loss_balance = self.ensemble_locals(torch.stack(loss_balance))

            loss_gan = self.args.w_gan * loss_gan
            loss_adv = self.args.w_adv * loss_adv
            loss_align = self.args.w_algn * loss_align
            loss_balance = self.args.w_baln * loss_balance

                # Final loss: L_align + L_local + L_adv (DRO) + L_balance
            loss = loss_gan + loss_adv + loss_align + loss_balance

        self.optim_g.zero_grad()
        if self.args.fp16:
            scaler_g = self.args.scaler_g
            scaler_g.scale(loss).backward()
            scaler_g.step(self.optim_g)
            scaler_g.update()
        else:
            loss.backward()
            self.optim_g.step()

        if self.writer is not None:
            self.writer.add_scalars('LossGen', {'loss_gan': loss_gan.item(),
                                                'loss_adv': loss_adv.item(),
                                                'loss_align': loss_align.item(),
                                                'loss_balance': loss_balance.item(),
                                                'loss_generator': loss.item()}, self.global_step)

    def forward_teacher_outs(self, images, localN=None):
        if localN is None:  # use central as teacher
            total_logits = self.netS(images).detach()
        else:  # update student
            # get local
            total_logits = []
            for n in localN:
                logits = self.netTS[n](images)
                total_logits.append(logits)
            total_logits = torch.stack(total_logits)  # nlocal*batch*ncls
            
        return total_logits

    def ensemble_locals(self, locals):
        """
        locals: (nlocal, batch, ncls) or (nlocal, batch/ncls) or (nlocal)
        """
        if len(locals.shape) == 3:
            localweight = self.localclsweight.unsqueeze(dim=1)  # nlocal*1*ncls
            ensembled = (locals * localweight).sum(dim=0)  # batch*ncls
        elif len(locals.shape) == 2:
            localweight = self.localweight[:, None]  # nlocal*1
            ensembled = (locals * localweight).sum(dim=0)  # batch/ncls
        elif len(locals.shape) == 1:
            ensembled = (locals * self.localweight).sum()  # 1
        return ensembled


    def update_netS_batch(self, selectN):
        for _ in range(5):
            with self.args.autocast():
                with torch.no_grad():
                    z = torch.randn(size=(self.args.batch_size, self.args.z_dim)).cuda()
                    syn_img_ori = self.netG(z)
                    syn_img = self.normalizer(syn_img_ori)
                    # apply pate DP
                    if self.args.use_pate:
                        clean_logits, logits_T = self.pate_teacher_outs(syn_img, self.args.lap_scale, selectN)
                    else:
                        logits_T = self.ensemble_locals(self.forward_teacher_outs(syn_img, selectN)) # logits_T.shape = [64,10]
                        
                
                logits_S = self.netS(syn_img.detach()) # logits_S.shape = [64,10]
                if self.args.use_pate and self.args.onehot_vote:
                    loss = self.criterion(logits_S, logits_T.detach())
                    # import pdb;pdb.set_trace()
                else:
                    loss = engine.criterions.kldiv(logits_S, logits_T.detach(), T=self.args.T)
                    # import pdb;pdb.set_trace()
                    
                loss *= self.args.w_dist

            self.optim_s.zero_grad()
            if self.args.fp16:
                scaler_s = self.args.scaler_s
                scaler_s.scale(loss).backward()
                scaler_s.step(self.optim_s)
                scaler_s.update()
            else:
                loss.backward()
                self.optim_s.step()

        if self.args.save_img:
            with self.args.autocast(), torch.no_grad():
                predict = logits_T[:self.args.batch_size].max(1)[1]
                idx = torch.argsort(predict)
                vis_images = syn_img_ori[idx]
                engine.utils.save_image_batch(self.args.normalizer(
                    self.local_dataloader[0].next()[0].cuda(), True), 'checkpoints/MosaicKD/real.png')
                engine.utils.save_image_batch(
                    vis_images, 'checkpoints/MosaicKD/syn.png')

        if self.writer is not None:
            self.writer.add_scalar(
                'LossDistill', loss.item(), self.global_step)

        

    def pate_teacher_outs(self, images, lap_scale, localN):
        """
        using pate
        """
        # use central as teacher
        if localN is None:  
            total_logits = self.netS(images).detach()
        else:  # update student
            # get local
            total_logits = []
            
            # one-hot不用再使用ensemble_locals整合，直接输出即可
            if self.args.onehot_vote:
                preds_sum = torch.zeros((self.args.batch_size, self.args.n_classes)).cuda()
                for n in localN:
                    logits = self.netTS[n](images) # [64,10]
                    total_logits.append(logits)
                # 1. 每个local对logits最大的类别投一票，其余投空票
                    pred = (logits >= logits.max(dim=1)[0].unsqueeze_(1)).type(torch.Tensor).cuda() # logits 最大值置1， 其余置0
                    if self.args.local_cls_weight:
                        # 每个class乘以local_n下对应的cls_weight
                        cls_weight = self.localclsweight[n].cuda() # [1,10]
                        w_preds = torch.mul(pred, cls_weight) # 带class权重的64（batchsize）*10(cls)投票阵
                    else:
                        w_preds = pred
                # 2. 全部w_preds加和，得到preds_sum为加权公投结果
                    if self.args.local_weight:
                        preds_sum += self.localweight[n]*w_preds
                    else:
                        preds_sum += w_preds
                # import pdb;pdb.set_trace()
                
                # 3. 得到onehot投票标签
                clean_logits = preds_sum
                if self.args.seed is not None and self.args.add_noise:
                    preds_weight = 1. # 当lap_scale越大，noise越小，对应预测结果需要乘以比例系数preds_weight
                    # np.random.seed(self.args.seed)
                    # torch.manual_seed(self.args.seed)
                    noise = torch.from_numpy(np.random.laplace(loc=0, scale=1/lap_scale, size=clean_logits.size())).cuda()
                else:
                    preds_weight = 1.
                    noise = 0.

                noisy_logits = clean_logits*preds_weight + noise
                # import pdb;pdb.set_trace()
                # one-hot编码的noisy_logits
                if self.args.max_vote:
                    noisy_logits = (noisy_logits >= noisy_logits.max(dim=1)[0].unsqueeze_(1)).type(torch.Tensor).cuda()
                else:
                    if self.args.local_cls_weight:
                        noisy_logits = (noisy_logits > 0.45).type(torch.Tensor).cuda()
                    elif self.args.local_weight:
                        # import pdb;pdb.set_trace()
                        raise NotImplementedError
                    else:
                        noisy_logits = (noisy_logits > self.args.N_PARTIES/2).type(torch.Tensor).cuda()
                

            else:
                print("Hi")
                for n in localN:
                    logits = self.netTS[n](images) # [64,10]
                    total_logits.append(logits)

                total_logits = torch.stack(total_logits)  # nlocal*batch*ncls
                clean_logits = total_logits
                noisy_logits = self.ensemble_locals(total_logits)
        
        return total_logits, noisy_logits.half()




def pate(data, netTD, lap_scale):
    """
    pate 方法来实现具备DP的教师预测结果
    """
    # data.size()[0] = batchsize
    # results为每位教师为每一张图片（数据）的投票结果
    results = torch.Tensor(len(netTD), data.size()[0]).type(torch.int64)
    for i in range(len(netTD)):
        # 每个教师都鉴别一遍输入数据
        output = netTD[i].forward(data)
        # output[64,10] -> pred[10,64] (>0.5的置1, 其余置0)
        pred = (output > 0.5).type(torch.Tensor).squeeze()
        # import pdb;pdb.set_trace()
        # 教师i的结果存入
        results[i] = pred
    
    clean_votes = torch.sum(results, dim=0).unsqueeze(1).type(torch.cuda.DoubleTensor)
    noise = torch.from_numpy(np.random.laplace(loc=0, scale=1/lap_scale, size=clean_votes.size())).cuda()
    noisy_results = clean_votes + noise
    noisy_labels = (noisy_results > len(netTD)/2).type(torch.cuda.DoubleTensor)

    return noisy_labels, clean_votes   


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if is_best:
        torch.save(state, os.path.join(save_dir, filename))
        print(f'[saved] ckpt saved to {os.path.join(save_dir, filename)}')
    else:
        torch.save(state, os.path.join(save_dir, 'latest.pth'))


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(current_lr, val_loader, model, criterion, args, current_epoch):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
        if args.rank <= 0:
            args.logger.info(
                ' [Eval] Epoch={current_epoch} Acc@1={top1.avg:.4f} Acc@5={top5.avg:.4f} Loss={losses.avg:.4f} Lr={lr:.5f}'
                .format(current_epoch=current_epoch, top1=top1, top5=top5, losses=losses, lr=current_lr))
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class MultiTeacher(OneTeacher):

    def gen_dataset(self, args):
        # 通过dirichlet分布分割数据，为每个local提供由num_classes类别组成的数量不一的数据
        # priv_train_data: 被分为N_PARTIES份数据，每份数据有num_classes, 服从dirichlet分布
        priv_train_data, ori_training_dataset, test_dataset = cifar.dirichlet_datasplit(
            args, privtype=args.dataset)

        local_dataset = []
        for n in range(self.args.N_PARTIES):
            # tr_dataset:
            # len(tr_dataset): 第n个local的训练数据量
            # tr_dataset[i] = (transform后的图像(3*32*32), label, idx)
            tr_dataset = cifar.Dataset_fromarray(priv_train_data[n]['x'],
                                                 priv_train_data[n]['y'],
                                                 train=True,
                                                 verbose=False)

            local_dataset.append(tr_dataset)
        # 直接从dataset中获取val_dataset
        num_classes, ori_training_dataset, val_dataset = registry.get_dataset(name=args.dataset,
                                                                              data_root=args.data_root)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers)

        local_datanum = np.zeros(self.args.N_PARTIES)
        local_cls_datanum = np.zeros((self.args.N_PARTIES, self.args.n_classes))
        for localid in range(self.args.N_PARTIES):
            # count
            local_datanum[localid] = priv_train_data[localid]['x'].shape[0]
            # class specific count
            for cls in range(self.args.n_classes):
                local_cls_datanum[localid, cls] = (
                    priv_train_data[localid]['y'] == cls).sum()

        assert sum(local_datanum) == 50000
        assert sum(sum(local_cls_datanum)) == 50000
        return val_loader, local_dataset, local_datanum, local_cls_datanum
