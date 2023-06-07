import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import random
from PIL import Image
import skvideo.io
import skimage
from skimage.measure import block_reduce
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Sampler, BatchSampler, RandomSampler, SubsetRandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VideoDataset(Dataset):
    def __init__(self, sample_ids, grid_file, agent_poss_file, actions_file, 
                 root_dir, model, start_sample, sample_interleave, 
                 ROWS=128, COLS=128,
                 seq_len=20, agent_type=0) -> None:
        super().__init__()
        self.start_sample = start_sample
        self.sample_interleave = sample_interleave
        self.seq_len = seq_len
        self.min_len = self.seq_len + 1
        self.ROWS = ROWS
        self.COLS = COLS
        self.SCALE = 30
        self.agent_type = torch.tensor(agent_type).to(device)
        self.gamma = torch.tensor([0.5, 0.9, 0.99]).to(device)
    
        ## load frames
        self.frames = []
        remains = []
        gif_files = [f'final-model-ppo-LunarLander-v2-{model}-{i}.mp4' for i in sample_ids]
        for i, (idx, file) in enumerate([*zip(sample_ids, gif_files)]):
            print(f'loading gif {idx} -- progress {i} / {len(sample_ids)}')
            frames = skvideo.io.vread(root_dir + file)[:-1]
            if len(frames) > self.start_sample + self.min_len * self.sample_interleave:
                remains.append(idx)
                frames = skimage.transform.resize(frames, (len(frames), ROWS, COLS, 3))
                self.frames.append(torch.tensor(np.moveaxis(frames, 3, 0),
                                                dtype=torch.float32, device=device))    

        ## load grids
        truth_data = np.load(root_dir+grid_file)
        truth_data = skimage.transform.resize(truth_data[remains], (len(remains), ROWS, COLS, 3))
        truth_data = np.moveaxis(truth_data, 3, 1)
        self.grids = torch.tensor(truth_data, dtype=torch.float32).to(device)
        agent_poss = np.load(root_dir+agent_poss_file)
        agent_poss = [agent_poss[_] for _ in [*agent_poss]]
        self.agent_poss = [torch.tensor(agent_poss[j], dtype=torch.float32, device=device) for j in remains]
        agent_actions = np.load(root_dir+actions_file)
        agent_actions = [agent_actions[_] for _ in [*agent_actions]]
        self.agent_actions = [torch.tensor(agent_actions[j], dtype=torch.uint8, device=device) for j in remains]

        ## sanity check
        assert len(self.frames) == len(self.grids), f"gifs and grids counts don't match ({len(self.frames)} and {len(self.grids)})"
        assert len(self.frames) == len(self.agent_poss), f"gifs and poss counts don't match ({len(self.frames)} and {len(self.agent_poss)})"
        assert len(self.frames) == len(self.agent_actions), f"gifs and actions counts don't match ({len(self.frames)} and {len(self.agent_actions)})"

        print('number of valid samples: ', len(remains))

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        sequence = self.frames[index]
        seq_len = sequence.shape[1]
        grid = self.grids[index]
        agent_poss = self.agent_poss[index]
        agent_poss[...,1] = (400-agent_poss[...,1]*self.SCALE) * self.ROWS/400
        agent_poss[...,0] = agent_poss[...,0]*self.SCALE * self.COLS/600
        agent_poss = agent_poss.type(torch.int64)
        actions = self.agent_actions[index]

        truncate_left = random.randrange(seq_len)
        truncate_right = min(seq_len-self.sample_interleave, truncate_left + self.seq_len*self.sample_interleave)
        if truncate_right - truncate_left < self.seq_len*self.sample_interleave:
            truncate_left = truncate_right - self.seq_len*self.sample_interleave
            assert truncate_left >= 0

        action = actions[truncate_right]

        target_sr = torch.zeros_like(sequence[:,0], device=device).permute(1,2,0)
                
        for step in range(truncate_right-self.sample_interleave, seq_len, self.sample_interleave):
            state = torch.zeros(self.ROWS, self.COLS, len(self.gamma), device=device)
            state[max(agent_poss[step,1]-2,0):min(agent_poss[step,1]+2, self.ROWS), 
                  max(agent_poss[step,0]-1,0):min(agent_poss[step,0]+2, self.ROWS), 
                      :] = 1
            target_sr += (self.gamma**((step-(truncate_right-self.sample_interleave))//self.sample_interleave)) * state
            # target_sr[idx,:,query_agent_pos[idx,step,0],query_agent_pos[idx,step,1]] += gamma**(step-length_recent[idx])
        target_sr[...,0] /= target_sr[...,0].sum()
        target_sr[...,1] /= target_sr[...,1].sum()
        target_sr[...,2] /= target_sr[...,2].sum()
        target_sr = target_sr.permute(2,1,0).to(device)
        
        sequence = sequence[:,truncate_left:truncate_right:self.sample_interleave]

        return (sequence, grid, action, self.agent_type, target_sr)