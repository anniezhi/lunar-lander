import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tensorflow as tf
import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset, TensorDataset
import argparse
import os
# import re
import numpy as np
# from PIL import Image
import cv2
import skimage
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import skvideo.io
# import pickle
from math import ceil
from model import *

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def dict2mdtable(d, key='Name', val='Value'):
    rows = [f'| {key} | {val} |']
    rows += ['|--|--|']
    rows += [f'| {k} | {v} |' for k, v in d.items()]
    return "  \n".join(rows)

parser = argparse.ArgumentParser()

parser.add_argument("--models", nargs='+', type=str, default=None,
                    help="names of the trained agent models")
parser.add_argument("--num-samples", type=int, default=500,
                    help="number of samples of each model to load")
parser.add_argument("--seq-length", type=int, default=20,
                    help="length of sequence for input")
parser.add_argument("--start-sample", type=int, default=0,
                    help="frame to start sampling")
parser.add_argument("--first-sample", type=int, default=0,
                    help="first sample id to load")
parser.add_argument("--sample-interleave", type=int, default=1,
                    help="interleave of sampling during taking frames from video")
parser.add_argument("--rows", type=int, default=64,
                    help="rows of resized image")
parser.add_argument("--cols", type=int, default=64,
                    help="cols of resized image")
parser.add_argument("--data-root-dir", type=str, default=None,
                    help="dir to data")
parser.add_argument("--save-dir", type=str, default=None,
                    help="dir to save trained model")
parser.add_argument("--epochs", type=int, default=100,
                    help="epoch number for trianing")
parser.add_argument("--epoch-size", type=int, default=20,
                    help="total episode number for training per epoch")
parser.add_argument("--batch-size", type=int, default=5,
                    help="batch size for trianing")
parser.add_argument("--log-interval", type=int, default=10,
                    help="number of epochs between logging decoder outputs")
parser.add_argument("--log-count", type=int, default=10,
                    help="number of samples to log in each epoch")
parser.add_argument("--lr", type=float, default=0.005,
                    help="learning rate (default: 0.005)")
parser.add_argument("--vae-weight", type=float, default=1.0,
                    help="VAE loss weight, set to 0 if not training")
parser.add_argument("--cl-weight", type=float, default=1.0,
                    help="CL loss weight, set to 0 if not training")
parser.add_argument("--kld-weight", type=float, default=0.00025,
                    help="kl-div weight in VAE loss (default: 0.00025)")
parser.add_argument("--kld-weight-anneal", type=str, default='none', choices=['none', 'simple', 'cyclic'],
                    help="kld weight annealing scheme, options [none, simple, cyclic]")
parser.add_argument("--cycles-anneal", type=int, default=2,
                    help="number of cycles for cyclic KLD loss annealing (default: 2)")
parser.add_argument("--prop-anneal", type=float, default=0.6,
                    help="proportion within a cycle to have increasing KLD loss weight (default: 0.6)")
parser.add_argument("--nce-weight", type=float, default=0.005,
                    help="nce weight in CL loss (default: 0.05)")
parser.add_argument("--loss-pos-factor", type=float, default=1.0,
                    help="Factor for pos loss (default: 1.0)")
parser.add_argument("--seed", type=int, default=1265,
                    help="random seed (default: 1265)")
parser.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam optimizer weight decay (default: 0.0)")
parser.add_argument("--scheduler-gamma", type=float, default=0.98,
                    help="lr scheduler gamma (default: 0.98)")
parser.add_argument("--nce-temp", type=float, default=0.5,
                    help="NCE loss temperature tau (default: 0.5)")
parser.add_argument("--recon-loss", type=str, default='mse', choices=['mse', 'huber'],
                    help="loss function for reconstruction, options [mse, huber]")
parser.add_argument("--milestones", nargs='+', type=int, default=[600],
                    help="milestones to decrease learning rate")
parser.add_argument("--continue-training", default=False, action='store_true',
                    help="continue training from last saved model")
parser.add_argument("--continue-from-best", default=False, action='store_true',
                    help="use best vae model when continue training if True, otherwise use last model")


if __name__ == "__main__":
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    # root_dir = os.getcwd() + '/lunar-lander/data_new/' + model + '/'
    if args.data_root_dir is None:
        data_root_dir = os.getcwd() + '/lunar-lander/data_videos/'
    else:
        data_root_dir = args.data_root_dir
    
    save_dir = os.getcwd() + '/lunar-lander/saved/' + args.save_dir + '/'
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    writer = SummaryWriter(log_dir=save_dir)
    args_dict = vars(args)
    writer.add_text("Hyperparameters", dict2mdtable(args_dict))

    ## Load data

    datasets = []
    ROWS, COLS = args.rows, args.cols

    for model_id, model in enumerate(args.models):
        root_dir = data_root_dir + model + '/'
        # actions_file = 'actions.npz'
        grid_file = 'env_grids.npy'
        agent_poss_file = 'agent_poss.npz'
        
        # truth_data = np.load(root_dir+grid_file)[:500]
        # agent_poss = np.load(root_dir+agent_poss_file)[:500]
        truth_data = np.load(root_dir+grid_file)
        agent_poss = np.load(root_dir+agent_poss_file)
        agent_poss = [agent_poss[_] for _ in [*agent_poss]]

        # agent_actions = np.load(root_dir+actions_file)
        # agent_actions = [agent_actions[_] for _ in [*agent_actions]]
    
        sample_ids = [i+args.first_sample for i in range(args.num_samples)]
        gif_files = [f'final-model-ppo-LunarLander-v2-{model}-{i}.mp4' for i in sample_ids]
        train_data_front = []
        train_data_mid = []
        train_data_back = []
        remains = []
        for idx, file in [*zip(sample_ids, gif_files)]:
            print(f'loading gif {idx} / {args.first_sample} - {args.first_sample+len(gif_files)}')
            # frames = []
            # im = Image.open(root_dir + 'gifs/' + file)
            # for t in range(im.n_frames):
            #     im.seek(t)
            #     tmp = im.convert()
            #     frames.append(skimage.transform.resize(np.asarray(tmp), (ROWS, COLS)))
            # if len(frames) >= 20:
            #     frames = np.asarray(frames[:20])
            #     train_data.append(frames[None,:])
            #     remains.append(idx)

            frames = skvideo.io.vread(root_dir + file)
            frames_len = len(frames)
            if frames_len >= args.start_sample + args.seq_length * args.sample_interleave:
                remains.append(idx)
                frames_front = frames[args.start_sample:args.start_sample+args.seq_length*args.sample_interleave:args.sample_interleave]
                frames_front = skimage.transform.resize(frames_front, (len(frames_front), ROWS, COLS, 3))
                train_data_front.append(frames_front[None, :])
                frames_back = frames[-args.seq_length*args.sample_interleave-1:-2:args.sample_interleave]
                frames_back = skimage.transform.resize(frames_back, (len(frames_back), ROWS, COLS, 3))
                train_data_back.append(frames_back[None, :])
                frames_mid = frames[int(frames_len/2-args.seq_length*args.sample_interleave/2):int(frames_len/2+args.seq_length*args.sample_interleave/2):args.sample_interleave]
                frames_mid = skimage.transform.resize(frames_mid, (len(frames_mid), ROWS, COLS, 3))
                train_data_mid.append(frames_mid[None, :])
                
        train_data_front = np.concatenate(train_data_front)
        train_data_front = np.moveaxis(train_data_front, 4, 1)
        train_data_mid = np.concatenate(train_data_mid)
        train_data_mid = np.moveaxis(train_data_mid, 4, 1)
        train_data_back = np.concatenate(train_data_back)
        train_data_back = np.moveaxis(train_data_back, 4, 1)
        train_data = np.concatenate([train_data_front, train_data_mid, train_data_back])

        truth_data = skimage.transform.resize(truth_data[remains], (len(remains), ROWS, COLS, 3))
        truth_data = np.moveaxis(truth_data, 3, 1)
        truth_data = np.tile(truth_data, (3, 1, 1, 1))
        
        # agent_poss = agent_poss[remains] / 255.0
        # agent_poss = np.moveaxis(agent_poss, 3, 1)
        agent_poss_front = np.stack([agent_poss[i][args.start_sample:args.start_sample+args.seq_length*args.sample_interleave:args.sample_interleave] for i in remains])
        agent_poss_mid = np.stack([agent_poss[i][int(len(agent_poss[i])/2-args.seq_length*args.sample_interleave/2):int(len(agent_poss[i])/2+args.seq_length*args.sample_interleave/2):args.sample_interleave] for i in remains])
        agent_poss_back = np.stack([agent_poss[i][-args.seq_length*args.sample_interleave-1:-2:args.sample_interleave] for i in remains])
        # convert to locations in resized image
        SCALE = 30
        agent_poss = np.concatenate([agent_poss_front, agent_poss_mid, agent_poss_back])
        agent_poss[...,1] = 400-agent_poss[...,1]*SCALE
        agent_poss[...,0] = agent_poss[...,0]*SCALE
        agent_poss[...,1] *= ROWS / 400
        agent_poss[...,0] *= COLS / 600
        agent_poss = agent_poss.astype(int)
        # add circle patches
        agent_poss_imgs = np.ones([len(truth_data), ROWS, COLS, 3])
        for i, poss in enumerate(agent_poss):
            for t in range(len(poss)):
                circle = cv2.circle(agent_poss_imgs[i], center=poss[t], 
                                    radius=3, color=(128/255, 102/255, 230/255),
                                    thickness=-1
                                    )
        agent_poss_imgs = np.moveaxis(agent_poss_imgs, 3, 1)

        # agent_actions_front = np.stack([agent_actions[i][args.start_sample:args.start_sample+args.seq_length*args.sample_interleave:args.sample_interleave] for i in remains])
        # agent_actions_mid = np.stack([agent_actions[i][int(len(agent_actions[i])/2-args.seq_length*args.sample_interleave/2):int(len(agent_actions[i])/2+args.seq_length*args.sample_interleave/2):args.sample_interleave] for i in remains])
        # agent_actions_back = np.stack([agent_actions[i][-args.seq_length*args.sample_interleave-1:-2:args.sample_interleave] for i in remains])
        # agent_actions = np.concatenate([agent_actions_front, agent_actions_mid, agent_actions_back])

        agent_label = np.array([model_id] * len(remains) * 3)

        print(f'model {model} is assigned id {model_id}')
        print(f'length of model dataset {len(remains) * 3}')
        
        datasets.append(TensorDataset(torch.tensor(train_data,dtype=torch.float32,device=device), 
                                      torch.tensor(truth_data,dtype=torch.float32,device=device), 
                                      torch.tensor(agent_poss_imgs,dtype=torch.float32,device=device), 
                                    #   torch.tensor(agent_actions,dtype=torch.float32,device=device), 
                                      torch.tensor(agent_label,device=device),
                                    #   torch.tensor(env_label,device=device)
                        ))
    
    dataset = ConcatDataset(datasets)
    print(f'length of full dataset {len(dataset)}')
    # sampler = RandomSampler(dataset, replacement=True, num_samples=args.epoch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)

    # model = RLNetwork(ROWS, COLS, 3, 4, 16)
    model = RLNetwork(ROWS, COLS, 3, 16, 4, 16)
    if args.continue_training:
        if args.continue_from_best:
            checkpoint = torch.load(save_dir + 'best_model.pt', map_location=torch.device(device))
        else:
            checkpoint = torch.load(save_dir + 'last_model.pt', map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_start = checkpoint['epoch']+1
        min_loss = checkpoint['loss']
    else:
        epoch_start = 0
        min_loss = 1e4
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    nce_loss = NCELoss(args.nce_temp)

    if args.recon_loss == 'mse':
        recon_loss_func = F.mse_loss
    elif args.recon_loss == 'huber':
        recon_loss_func = F.huber_loss

    for epoch in range(epoch_start, epoch_start+args.epochs):

        # contrast_loss_env = 0
        contrast_loss_agent = 0
        recon_loss_grid = 0
        recon_loss_pos = 0
        kl_loss = 0

        logging = False

        if (epoch+1) % args.log_interval == 0:
            logging = True
            counter_log = 0
            log_grids_truth = []
            log_grids_recon = []

            log_poss_truth = []
            log_poss_recon = []

            # log_actions_truth = []
            # log_actions_recon = []

            log_embeds_enc = []
            log_classes_agent = []
            log_classes_env = []

        for batch_id, batch_data in enumerate(dataloader):
            
            # seqs_train, grids_truth, poss_truth, agent_actions, agent_class = batch_data
            seqs_train, grids_truth, poss_truth, agent_class = batch_data
            grids_recon, poss_recon, mu, logvar, projections = model(seqs_train)

            recon_loss_grid += recon_loss_func(grids_truth, grids_recon)
            recon_loss_pos += recon_loss_func(poss_truth, poss_recon)
            
            # contrast_loss_env += nce_loss(projections, env_class.tolist())
            contrast_loss_agent += nce_loss(projections, agent_class.tolist())

            kl_loss += -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
        
            ## log recon to tensorboard
            if logging and counter_log < args.log_count:
                log_grids_truth.append(grids_truth[:args.log_count-counter_log].detach())
                log_grids_recon.append(grids_recon[:args.log_count-counter_log].detach())

                log_poss_truth.append(poss_truth[:args.log_count-counter_log].detach())
                log_poss_recon.append(poss_recon[:args.log_count-counter_log].detach())

                log_embeds_enc.append(projections[:args.log_count-counter_log].detach())

                # log_classes_env.append(env_class[:args.log_count-counter_log].detach())
                log_classes_agent.append(agent_class[:args.log_count-counter_log].detach())

                counter_log += min(len(grids_recon), args.log_count-counter_log)

        contrast_loss_agent /= (batch_id+1)
        # contrast_loss_env /= (batch_id+1)
        # cl_loss = args.nce_weight * (contrast_loss_agent + contrast_loss_env)
        cl_loss = args.nce_weight * contrast_loss_agent
        cl_loss *= args.cl_weight

        recon_loss_grid /= (batch_id+1)
        recon_loss_pos /= (batch_id+1)
        recon_loss = (2-args.loss_pos_factor) * recon_loss_grid + args.loss_pos_factor * recon_loss_pos

        kl_loss /= (batch_id+1)
        if args.kld_weight_anneal == 'simple':
            anneal_weight = epoch/args.epochs
        elif args.kld_weight_anneal == 'none':
            anneal_weight = 1.0
        elif args.kld_weight_anneal == 'cyclic':
            tau = (epoch % ceil(args.epochs / args.cycles_anneal)) / (args.epochs / args.cycles_anneal)
            if tau <= args.prop_anneal:
                anneal_weight = tau / args.prop_anneal
            else:
                anneal_weight = 1.0
        kl_loss = args.kld_weight * anneal_weight * kl_loss

        vae_loss = recon_loss + kl_loss
        vae_loss *= args.vae_weight

        loss = cl_loss + vae_loss
        # loss = vae_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        ## log numbers to tensorboard
        print('epoch {} loss {:.4f} (rec_env {:.4f} rec_pos {:.4f} kl {:.4f} cl {:.4f} lr {:.4f})'.format(epoch, loss.item(), 
                                                                                                        args.vae_weight*recon_loss_grid.item(),
                                                                                                        args.vae_weight*recon_loss_pos.item(),
                                                                                                        args.vae_weight*kl_loss.item(),
                                                                                                        cl_loss.item(),
                                                                                                        scheduler.get_last_lr()[0]
                                                                                                        ))
        writer.add_scalar('loss_total/train', loss.item(), epoch)
        writer.add_scalar('loss_cl/train', cl_loss.item(), epoch)
        writer.add_scalar('loss_vae/train', vae_loss.item(), epoch)
        writer.add_scalar('loss_recon_grid/train', args.vae_weight*recon_loss_grid.item(), epoch)
        writer.add_scalar('loss_recon_pos/train', args.vae_weight*recon_loss_pos.item(), epoch)
        writer.add_scalar('loss_kl/train', args.vae_weight*kl_loss.item(), epoch)
        writer.add_scalar('lr/train', scheduler.get_last_lr()[0], epoch)

        ## log outputs to tensorboard
        if logging:
            writer.add_images('truth_envs/train', torch.concat(log_grids_truth), epoch, dataformats='NCWH')
            writer.add_images('recon_envs/train', torch.concat(log_grids_recon), epoch, dataformats='NCWH')
            writer.add_images('truth_traces/train', torch.concat(log_poss_truth), epoch, dataformats='NCWH')
            writer.add_images('recon_traces/train', torch.concat(log_poss_recon), epoch, dataformats='NCWH')
            writer.add_embedding(torch.concat(log_embeds_enc), metadata=torch.concat(log_classes_agent).tolist(), global_step=epoch, tag='enc_embeds_agent_train')
            # writer.add_embedding(torch.concat(log_embeds_enc), metadata=torch.concat(log_classes_env).tolist(), global_step=epoch, tag='enc_embeds_env_train')

        # save checkpoint
        if loss <= min_loss:
            min_loss = loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
            }, save_dir+'best_model.pt')

    ## save checkpoint
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
    }, save_dir+'last_model.pt')

    writer.close()
