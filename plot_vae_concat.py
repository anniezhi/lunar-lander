import torch
import torch.nn as nn
import tensorflow as tf
import tensorboard as tb
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import numpy as np
import random
import skimage
import cv2
from PIL import Image
import skvideo.io
from model import *

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

parser = argparse.ArgumentParser()

parser.add_argument("--vae-model", type=str, default=None,
                    help="names of the trained RLRL VAE models")
parser.add_argument("--agent-model", type=str, default=None,
                    help="names of the trained agent model")
parser.add_argument("--data-root-dir", type=str, default=None,
                    help="dir to data")
parser.add_argument("--mode", type=str, default='train', choices=['train', 'test'],
                    help="train mode and test mode load different videos")
parser.add_argument("--best-model", default=False, action='store_true',
                    help="use best vae model if True, otherwise use last model")
parser.add_argument("--first-sample", type=int, default=0,
                    help="first sample id to load")
parser.add_argument("--num-samples", type=int, default=100,
                    help="number of samples to load")
parser.add_argument("--start-sample", type=int, default=0,
                    help="frame to start sampling")
parser.add_argument("--sample-interleave", type=int, default=1,
                    help="interleave of sampling during taking frames from video")
parser.add_argument("--seq-length", type=int, default=20,
                    help="length of sequence for input")
parser.add_argument("--seed", type=int, default=1265,
                    help="random seed (default: 1265)")

if __name__ == "__main__":
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    if args.data_root_dir is None:
        data_root_dir = os.getcwd() + '/lunar-lander/data_videos/'
    else:
        data_root_dir = args.data_root_dir
    data_dir = data_root_dir + args.agent_model + '/'
    # data_dir = '/cluster/work/hilliges/xiazhi' + '/lunar-lander/data_videos/' + model + '/'

    model_dir = os.getcwd() + '/lunar-lander/saved/' + args.vae_model + '/'
    if not os.path.exists(model_dir):
        print(f'VAE model {model_dir} not found')
    save_dir = os.getcwd() + '/lunar-lander/plots_vae/' + args.vae_model + '/' + args.agent_model + '/'
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    print('VAE model: ', args.vae_model)
    print('agent model: ', args.agent_model)

    ROWS, COLS = 128, 128

    ## Load data
    actions_file = 'actions.npz'
    grid_file = 'env_grids.npy'
    agent_poss_file = 'agent_poss.npz'

    truth_data = np.load(data_dir+grid_file)
    agent_poss = np.load(data_dir+agent_poss_file)
    agent_poss = [agent_poss[_] for _ in [*agent_poss]]
    
    # agent_actions = np.load(data_dir+actions_file)
    # agent_actions = [agent_actions[_] for _ in [*agent_actions]]

    # sample_ids = random.sample(range(len(truth_data)), int(args.num_samples*1.5))
    sample_ids = [i+args.first_sample for i in range(args.num_samples)]
    gif_files = [f'final-model-ppo-LunarLander-v2-{args.agent_model}-{i}.mp4' for i in sample_ids]
    test_data = []
    remains = []
    for idx, file in [*zip(sample_ids, gif_files)]:
    # for idx, file in enumerate(gif_files):
        frames = skvideo.io.vread(data_dir + file)
        if len(frames) >= args.start_sample + args.seq_length * args.sample_interleave:
            # frames = frames[args.start_sample:args.start_sample+args.seq_length*args.sample_interleave:args.sample_interleave]
            frames = frames[args.start_sample::args.sample_interleave]
            frames = skimage.transform.resize(frames, (len(frames), ROWS, COLS, 3))
            test_data.append(np.moveaxis(frames[None, :],4,1))
            remains.append(idx)
    
    remains = remains[:args.num_samples]
    print('number of valid samples: ', len(remains))

    test_data = test_data[:args.num_samples]
    # test_data = np.concatenate(test_data)
    # test_data = np.moveaxis(test_data, 4, 1)

    truth_data = skimage.transform.resize(truth_data[remains], (len(remains), ROWS, COLS, 3))
    truth_data = np.moveaxis(truth_data, 3, 1)
    
    # agent_poss = np.stack([agent_poss[i][args.start_sample:args.start_sample+args.seq_length*args.sample_interleave:args.sample_interleave] for i in remains])
    agent_poss = [agent_poss[i][args.start_sample::args.sample_interleave] for i in remains]
    # convert to locations in resized image
    # SCALE = 30
    # agent_poss[...,1] = 400-agent_poss[...,1]*SCALE
    # agent_poss[...,0] = agent_poss[...,0]*SCALE
    # agent_poss[...,1] *= ROWS / 400
    # agent_poss[...,0] *= COLS / 600
    # agent_poss = agent_poss.astype(int)
    # # add circle patches
    # agent_poss_imgs = np.ones([len(truth_data), ROWS, COLS, 3])
    # for i, poss in enumerate(agent_poss):
    #         for t in range(len(poss)):
    #             circle = cv2.circle(agent_poss_imgs[i], center=poss[t], 
    #                                 radius=3, color=(128/255, 102/255, 230/255),
    #                                 thickness=-1
    #                                 )
    # agent_poss_imgs = np.moveaxis(agent_poss_imgs, 3, 1)
    # agent_actions = np.stack([agent_actions[i][args.start_sample:args.start_sample+args.seq_length*args.sample_interleave:args.sample_interleave] for i in remains])    

    # dataset = TensorDataset(torch.tensor(test_data,dtype=torch.float32,device=device), 
    #                         torch.tensor(truth_data,dtype=torch.float32,device=device), 
    #                         torch.tensor(agent_poss_imgs,dtype=torch.float32,device=device),
    #                         # torch.tensor(agent_actions,dtype=torch.float32,device=device),
    #                         torch.tensor(remains,dtype=torch.int,device=device)
    #                         )

    # dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)

    ## load trained VAE model
    model_VAE = RLNetwork(ROWS, COLS, 3, 16, 4, 16)
    if args.best_model:
        checkpoint = torch.load(model_dir + 'best_model.pt', map_location=torch.device(device))
    else:
        checkpoint = torch.load(model_dir + 'last_model.pt', map_location=torch.device(device))
    model_VAE.load_state_dict(checkpoint['model_state_dict'])
    model_VAE.to(device)
    # model_VAE.eval()
    
    criterion = nn.MSELoss(reduction='sum')

    acc = 0
    recon_loss_grid = 0
    recon_loss_pos = 0
    log_embeds_enc = []

    # for batch_id, batch_data in enumerate(dataloader):
    for id in range(len(truth_data)):
        
        seq_test = test_data[id]
        grid_truth = truth_data[id]
        agent_pos = agent_poss[id]
        remain = remains[id]

        SCALE = 30
        agent_pos[...,1] = 400-agent_pos[...,1]*SCALE
        agent_pos[...,0] = agent_pos[...,0]*SCALE
        agent_pos[...,1] *= ROWS / 400
        agent_pos[...,0] *= COLS / 600
        agent_pos = agent_pos.astype(int)

        seqs_test = []
        agent_pos_repeat = []

        for t in range(seq_test.shape[2]//args.seq_length):
            seqs_test.append(seq_test[:,:,t*args.seq_length:(t+1)*args.seq_length])
            agent_pos_img = np.ones([ROWS, COLS, 3])
            for pos in agent_pos[t*args.seq_length:(t+1)*args.seq_length]:
                circle = cv2.circle(agent_pos_img, center=pos, 
                                    radius=3, color=(128/255, 102/255, 230/255),
                                    thickness=-1
                                    )
            agent_pos_repeat.append(np.moveaxis(agent_pos_img[None], 3, 1))

        seqs_test = torch.tensor(np.concatenate(seqs_test), dtype=torch.float32, device=device)
        grids_truth = torch.tensor(grid_truth[None].repeat(t+1, axis=0), dtype=torch.float32, device=device)
        agent_pos_repeat = torch.tensor(np.concatenate(agent_pos_repeat), dtype=torch.float32, device=device)

        grids_recon, agent_poss_recon, mu, logvar, projections = model_VAE(seqs_test)

        recon_loss_grid += criterion(grids_truth, grids_recon) / (t+1)
        recon_loss_pos += criterion(agent_pos_repeat, agent_poss_recon) / (t+1)
        
        ## save images
        grids_recon = grids_recon.detach().permute(0,2,3,1).numpy()
        grids_recon = np.around(grids_recon * 255).astype(np.uint8)
        
        grids_truth = grids_truth.permute(0,2,3,1).numpy()
        grids_truth = np.around(grids_truth * 255).astype(np.uint8)
        
        agent_poss_recon = agent_poss_recon.detach().permute(0,2,3,1).numpy()
        agent_poss_recon = np.around(agent_poss_recon * 255).astype(np.uint8)
        
        agent_pos_repeat = agent_pos_repeat.permute(0,2,3,1).numpy()
        agent_pos_repeat = np.around(agent_pos_repeat * 255).astype(np.uint8)

        grids_recon_stack = np.mean(grids_recon, axis=0).astype(np.uint8)
        agent_poss_recon_stack = np.sum(agent_poss_recon, axis=0).astype(np.uint8)

        grid_recon_im = Image.fromarray(grids_recon_stack)
        grid_recon_im.save(save_dir+ f'grid_recon_{remain}_stacked.jpg')

        grid_truth_im = Image.fromarray(grids_truth[0])
        grid_truth_im.save(save_dir+ f'grid_truth_{remain}_stacked.jpg')

        poss_recon_im = Image.fromarray(agent_poss_recon_stack)
        poss_recon_im.save(save_dir+ f'poss_recon_{remain}_stacked.jpg')
        
        poss_truth_im = Image.fromarray(agent_pos_repeat[0])
        poss_truth_im.save(save_dir+ f'poss_truth_{remain}_stacked.jpg')

        # for i in range(len(grids_recon)):
        #     grid_recon_im = Image.fromarray(grids_recon[i])
        #     grid_recon_im.save(save_dir+ f'grid_recon_{ids[i]}.jpg')

        #     grid_truth_im = Image.fromarray(grids_truth[i])
        #     grid_truth_im.save(save_dir+ f'grid_truth_{ids[i]}.jpg')

        #     poss_recon_im = Image.fromarray(agent_poss_recon[i])
        #     poss_recon_im.save(save_dir+ f'poss_recon_{ids[i]}.jpg')
            
        #     poss_truth_im = Image.fromarray(agent_poss[i])
        #     poss_truth_im.save(save_dir+ f'poss_truth_{ids[i]}.jpg')
            
        ## log embeddings
        log_embeds_enc.append(projections.mean(0).detach().numpy())

    ## save embeddings
    embeds = np.concatenate(log_embeds_enc)
    np.save(save_dir+'embeddings-'+args.mode+'mixed.npy', embeds)

    recon_loss_grid /= len(truth_data)
    recon_loss_pos /= len(truth_data)
    # acc = acc.float() / len(dataset)

    print('recon MSE loss grid: ', recon_loss_grid)
    print('recon MSE loss pos: ', recon_loss_pos)

    ## log key information to txt
    with open(save_dir+'info-'+args.mode+'-mixed.txt', 'w') as f:
        f.write(f'VAE model: {args.vae_model}\n')
        f.write(f'agent model: {args.agent_model}\n')
        f.write(f'number of valid samples: {len(remains)}\n')
        f.write(f'recon MSE loss grid: {recon_loss_grid}\n')
        f.write(f'recon MSE loss pos: {recon_loss_pos}')