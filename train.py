import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tensorflow as tf
import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset, TensorDataset
import argparse
import os
import re
import numpy as np
from PIL import Image
import cv2
import skimage
import pickle
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
parser.add_argument("--save-dir", type=str, default=None,
                    help="dir to save trained model")
parser.add_argument("--epochs", type=int, default=100,
                    help="epoch number for trianing")
parser.add_argument("--epoch-size", type=int, default=20,
                    help="total episode number for trianing per epoch")
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


if __name__ == "__main__":
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    save_dir = os.getcwd() + '/lunar-lander/saved/' + args.save_dir + '/'
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    writer = SummaryWriter(log_dir=save_dir)
    args_dict = vars(args)
    writer.add_text("Hyperparameters", dict2mdtable(args_dict))

    ## Load data

    datasets = []
    ROWS, COLS = 128, 128

    # props_agent = set()
    # props_env = set()
    # for model in args.models:
    #     agent_v = int(re.search(r"(?<=v)\d+", model).group())
    #     props_agent.add(agent_v)
    #     env_N = int(re.search(r"(?<=N)\d+", model).group())
    #     props_env.add(env_N)
    # labels_dict_agent = dict(zip(props_agent, list(range(len(props_agent)))))
    # labels_dict_env = dict(zip(props_env, list(range(len(props_env)))))

    for model in args.models:
        root_dir = os.getcwd() + '/lunar-lander/data_new/' + model + '/'
        actions_file = 'actions.pkl'
        # grid_file = 'env_grids_rgb_double.npy'
        # agent_poss_file = 'agent_poss_trimmed.npy'

        # agent_v = int(re.search(r"(?<=v)\d+", model).group())
        # env_N = int(re.search(r"(?<=N)\d+", model).group())
        
        # truth_data = np.load(root_dir+grid_file)[:500]
        # agent_poss = np.load(root_dir+agent_poss_file)[:500]

        with open(root_dir+actions_file, 'rb') as f:
            agent_actions = pickle.load(f)
    
        gif_files = [f'{i}.gif' for i in range(args.num_samples)]
        train_data = []
        remains = []
        for idx, file in enumerate(gif_files):
            print(f'loading gif {idx} / {len(gif_files)}')
            frames = []
            im = Image.open(root_dir + 'gifs/' + file)
            for t in range(im.n_frames):
                im.seek(t)
                tmp = im.convert()
                frames.append(skimage.transform.resize(np.asarray(tmp), (ROWS, COLS)))
            if len(frames) >= 20:
                frames = np.asarray(frames[:20])
                train_data.append(frames[None,:])
                remains.append(idx)
        
        train_data = np.concatenate(train_data)
        train_data = np.moveaxis(train_data, 4, 1)
        truth_data = train_data.mean(2)
        # agent_poss = agent_poss[remains] / 255.0
        # agent_poss = np.moveaxis(agent_poss, 3, 1)
        agent_actions = np.stack([agent_actions[i][:20] for i in remains])
        # agent_label = np.array([labels_dict_agent[agent_v]] * len(remains))
        agent_label = np.array([0]*len(train_data))
        # env_label = np.array([labels_dict_env[env_N]] * len(remains))

        datasets.append(TensorDataset(torch.tensor(train_data,dtype=torch.float32,device=device), 
                                                 torch.tensor(truth_data,dtype=torch.float32,device=device), 
                                                #  torch.tensor(agent_poss,dtype=torch.float32,device=device), 
                                                 torch.tensor(agent_actions,dtype=torch.float32,device=device), 
                                                 torch.tensor(agent_label,device=device),
                                                #  torch.tensor(env_label,device=device)
                        ))
    
    dataset = ConcatDataset(datasets)
    print(f'length of dataset {len(dataset)}')
    sampler = RandomSampler(dataset, replacement=True, num_samples=args.epoch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, sampler=sampler)

    model = RLNetwork(ROWS, COLS, 3, 4, 16)
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600], gamma=0.5)

    # nce_loss = NCELoss(args.nce_temp)

    if args.recon_loss == 'mse':
        recon_loss_func = F.mse_loss
    elif args.recon_loss == 'huber':
        recon_loss_func = F.huber_loss

    min_loss = 1e4

    for epoch in range(args.epochs):

        # contrast_loss_env = 0
        # contrast_loss_agent = 0
        recon_loss_grid = 0
        recon_loss_pos = 0
        kl_loss = 0

        logging = False

        if (epoch+1) % args.log_interval == 0:
            logging = True
            counter_log = 0
            log_grids_truth = []
            log_grids_recon = []

            # log_poss_truth = []
            # log_poss_recon = []

            log_actions_truth = []
            log_actions_recon = []

            log_embeds_enc = []
            log_classes_agent = []
            log_classes_env = []

        for batch_id, batch_data in enumerate(dataloader):
            
            seqs_train, grids_truth, agent_actions, agent_class = batch_data
            grids_recon, poss_recon, mu, logvar, projections = model(seqs_train)

            recon_loss_grid += recon_loss_func(grids_truth, grids_recon)
            # recon_loss_pos += recon_loss_func(agent_poss, poss_recon)
            
            # contrast_loss_env += nce_loss(projections, env_class.tolist())
            # contrast_loss_agent += nce_loss(projections, agent_class.tolist())

            kl_loss += -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
        
            ## log recon to tensorboard
            if logging and counter_log < args.log_count:
                log_grids_truth.append(grids_truth[:args.log_count-counter_log].detach())
                log_grids_recon.append(grids_recon[:args.log_count-counter_log].detach())

                # log_poss_truth.append(agent_poss[:args.log_count-counter_log].detach())
                # log_poss_recon.append(poss_recon[:args.log_count-counter_log].detach())

                log_embeds_enc.append(projections[:args.log_count-counter_log].detach())

                # log_classes_env.append(env_class[:args.log_count-counter_log].detach())
                log_classes_agent.append(agent_class[:args.log_count-counter_log].detach())

                counter_log += min(len(grids_recon), args.log_count-counter_log)

        # contrast_loss_agent /= (batch_id+1)
        # contrast_loss_env /= (batch_id+1)
        # cl_loss = args.nce_weight * (contrast_loss_agent + contrast_loss_env)
        # cl_loss *= args.cl_weight

        recon_loss_grid /= (batch_id+1)
        recon_loss_pos /= (batch_id+1)
        recon_loss = recon_loss_grid + recon_loss_pos

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

        # loss = cl_loss + vae_loss
        loss = vae_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        ## log numbers to tensorboard
        print('epoch {} loss {:.4f} (rec_env {:.4f} kl {:.4f} lr {:.4f})'.format(epoch, loss.item(), 
                                                                                                        args.vae_weight*recon_loss_grid.item(),
                                                                                                        args.vae_weight*kl_loss.item(),
                                                                                                        scheduler.get_last_lr()[0]
                                                                                                        ))
        writer.add_scalar('loss_total/train', loss.item(), epoch)
        # writer.add_scalar('loss_cl/train', cl_loss.item(), epoch)
        writer.add_scalar('loss_vae/train', vae_loss.item(), epoch)
        writer.add_scalar('loss_recon_grid/train', args.vae_weight*recon_loss_grid.item(), epoch)
        # writer.add_scalar('loss_recon_pos/train', args.vae_weight*recon_loss_pos.item(), epoch)
        writer.add_scalar('loss_kl/train', args.vae_weight*kl_loss.item(), epoch)
        writer.add_scalar('lr/train', scheduler.get_last_lr()[0], epoch)

        ## log outputs to tensorboard
        if logging:
            writer.add_images('truth_envs/train', torch.concat(log_grids_truth), epoch, dataformats='NCWH')
            writer.add_images('recon_envs/train', torch.concat(log_grids_recon), epoch, dataformats='NCWH')
            # writer.add_images('truth_traces/train', torch.concat(log_poss_truth), epoch, dataformats='NCWH')
            # writer.add_images('recon_traces/train', torch.concat(log_poss_recon), epoch, dataformats='NCWH')
            writer.add_embedding(torch.concat(log_embeds_enc), metadata=torch.concat(log_classes_agent).tolist(), global_step=epoch, tag='enc_embeds_agent_train')
            # writer.add_embedding(torch.concat(log_embeds_enc), metadata=torch.concat(log_classes_env).tolist(), global_step=epoch, tag='enc_embeds_env_train')

        ## save checkpoint
        if loss <= min_loss:
            min_loss = loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
            }, save_dir+'best_model.pt')

    writer.close()
