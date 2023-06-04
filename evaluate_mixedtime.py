import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import tensorflow as tf
import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import argparse
import os
import re
import pickle
from PIL import Image
from model import *
from data_utils import *

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def ce_soft_label(target, pred):
    return -(target * torch.log(pred)).sum()

parser = argparse.ArgumentParser()

parser.add_argument("--vae-model", type=str, default=None,
                    help="names of the trained RLRL VAE models")
parser.add_argument("--agent-models", nargs='+', type=str, default=None,
                    help="names of the trained agent models")
parser.add_argument("--data-root-dir", type=str, default=None,
                    help="dir to data")
parser.add_argument("--first-sample", type=int, default=0,
                    help="first sample id to load")
parser.add_argument("--num-samples", type=int, default=100,
                    help="number of samples to load")
parser.add_argument("--val-ratio", type=float, default=0.2,
                    help="Ratio of vaildation loss (default: 0.2)")
parser.add_argument("--seq-length", type=int, default=20,
                    help="length of sequence for input")
parser.add_argument("--best-model", default=False, action='store_true',
                    help="use best vae model if True, otherwise use last model")
parser.add_argument("--save-model", default=False, action='store_true',
                    help="save model after training")
parser.add_argument("--downstream-model", type=str, default=None,
                    help="names of the downstream RLRL evaluation models")
parser.add_argument("--weight-loss-goal", type=float, default=1.0,
                    help="Weight of loss term goal (default: 1.0)")
parser.add_argument("--weight-loss-v", type=float, default=1.0,
                    help="Weight of loss term agent v (default: 1.0)")
parser.add_argument("--weight-loss-env", type=float, default=1.0,
                    help="Weight of loss term env (default: 1.0)")
parser.add_argument("--weight-loss-action", type=float, default=1.0,
                    help="Weight of loss term action (default: 1.0)")
parser.add_argument("--weight-loss-sr", type=float, default=1.0,
                    help="Weight of loss term SR (default: 1.0)")
parser.add_argument("--epochs", type=int, default=100,
                    help="epoch number for trianing")
parser.add_argument("--batch-size", type=int, default=5,
                    help="batch size for trianing")
parser.add_argument("--log-interval", type=int, default=10,
                    help="number of epochs between logging decoder outputs")
parser.add_argument("--log-count", type=int, default=8,
                    help="number of samples to log in each epoch")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.005)")
parser.add_argument("--lr-milestones", nargs='+', type=int, default=None,
                    help="milestones to half the learning rates")
parser.add_argument("--seed", type=int, default=1265,
                    help="random seed (default: 1265)")
parser.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam optimizer weight decay (default: 0.0)")
parser.add_argument("--scheduler-gamma", type=float, default=0.5,
                    help="lr scheduler gamma (default: 0.5)")

if __name__ == "__main__":
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    model_dir = os.getcwd() + '/RLRL/saved/' + args.vae_model + '/'
    if not os.path.exists(model_dir):
        print(f'VAE model {model_dir} not found')
    save_dir = os.getcwd() + '/RLRL/saved_downstream/' + args.downstream_model + '/'
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    writer = SummaryWriter(log_dir=save_dir)
    args_dict = vars(args)
    writer.add_text("Hyperparameters", dict2mdtable(args_dict))

    ## Load data

    datasets = []

    labels_dict_agent_v = {3: 0, 5: 1, 7: 2}
    labels_dict_agent_goal = {'green': 0, 'yellow': 1, 'purple': 2}
    labels_dict_env = {1: 0, 2: 1, 3: 2}
    writer.add_text("Labels: agent speed", dict2mdtable(labels_dict_agent_v))
    writer.add_text("Labels: goal", dict2mdtable(labels_dict_agent_goal))
    writer.add_text("Labels: environment", dict2mdtable(labels_dict_env))

    gamma = np.array([0.5, 0.9, 0.99])

    if args.data_root_dir is None:
        data_root_dir = os.getcwd() + '/RLRL/data/'
    else:
        data_root_dir = args.data_root_dir

    seq_length_initial = args.seq_length + 1

    for model in args.agent_models:
        root_dir = data_root_dir + model + '/'
        grid_file = 'env_grids_rgb_double.npy'
        # agent_poss_file = 'agent_poss_trimmed.npy'

        agent_v = int(re.search(r"(?<=v)\d+", model).group())
        agent_goal = re.search(r"goal(\w+)", model).group(1)
        env_N = int(re.search(r"(?<=N)\d+", model).group())
        
        truth_data = np.load(root_dir+grid_file)
        # agent_poss = np.load(root_dir+agent_poss_file)[:500]
    
        sample_ids = [i+args.first_sample for i in range(args.num_samples)]

        gif_files = [f'{i}-double.gif' for i in sample_ids]
        test_data = []
        target_srs = []
        remains = []

        datasets.append(VideoDataset(sample_ids, grid_file, root_dir, args.seq_length,
                                     labels_dict_agent_v[agent_v], labels_dict_agent_goal[agent_goal],
                                     labels_dict_env[env_N]))

        # for idx, file in [*zip(sample_ids, gif_files)]:
        #     frames = []
        #     im = Image.open(root_dir + 'gifs/' + file)
        #     for t in range(im.n_frames):
        #         im.seek(t)
        #         tmp = im.convert()
        #         frames.append(np.asarray(tmp))
        #     frames_len = len(frames)
        #     if frames_len > seq_length_initial:
        #         remains.append(idx)
                # frames = np.asarray(frames[:args.seq_length])
                # test_data.append(frames[None,:])

                # if args.time == 'front':
                #     truncate_start = 0
                # elif args.time == 'back':
                #     truncate_start = frames_len - seq_length_initial
                # elif args.time == 'mid':
                #     truncate_start = int(frames_len/2-seq_length_initial/2)
                # else:
                #     left = max(0, int(args.time_ratio * frames_len - seq_length_initial/2))
                #     right = min(frames_len, int(args.time_ratio * frames_len + seq_length_initial/2))
                #     if right - left < seq_length_initial:
                #         right = min(frames_len, left + seq_length_initial)
                #         left = max(0, right - seq_length_initial)
                #     truncate_start = left

                ## sr
                # target_sr = np.zeros_like(frames[0], dtype=np.float64)
                # for step in range(truncate_start+seq_length_initial, frames_len):
                #     state = (frames[step] - truth_data[idx])[:,:,0,None] / 255.0
                #     state = np.repeat(state,len(gamma),-1)
                #     target_sr += (gamma**(step-(truncate_start+seq_length_initial-1))) * state
                #     # target_sr[idx,:,query_agent_pos[idx,step,0],query_agent_pos[idx,step,1]] += gamma**(step-length_recent[idx])
                # target_sr[...,0] /= np.sum(target_sr[...,0])
                # target_sr[...,1] /= np.sum(target_sr[...,1])
                # target_sr[...,2] /= np.sum(target_sr[...,2])
                # target_sr = np.transpose(target_sr, (2,0,1))
                # target_srs.append(target_sr[None,:])

                # test_data.append(np.asarray(frames[truncate_start:truncate_start+seq_length_initial])[None,:])
                # test_data.append(np.asarray(frames))

        # remains = remains[:args.num_samples]
        # test_data = test_data[:args.num_samples]
        # print('number of valid samples: ', len(remains))

        # truth_data = truth_data[remains] / 255.0
        # truth_data = np.moveaxis(truth_data, 3, 1)
        # test_data = np.concatenate(test_data) / 255.0
        # test_data = np.moveaxis(test_data, 4, 1)
        # target_srs = np.concatenate(target_srs)
        # truth_repeat = truth_data[:,:,None,:,:].repeat(seq_length_initial, axis=2)
        # agent_poss_seq = test_data - truth_repeat
        # agent_poss = agent_poss_seq.max(2)

        # # agent pos at next step after seq
        # agent_pos_next = agent_poss_seq[:,0,-1].reshape(agent_poss_seq.shape[0],-1).argmax(1)
        # y_pos_next, x_pos_next = np.divmod(agent_pos_next, 22)
        # # agent pos at last step in seq
        # agent_pos_last = agent_poss_seq[:,0,-2].reshape(agent_poss_seq.shape[0],-1).argmax(1)
        # y_pos_last, x_pos_last = np.divmod(agent_pos_last, 22)

        # action based on pos change
        # truth_actions = (x_pos_next > x_pos_last) * 1 + (x_pos_next < x_pos_last) * 2 + \
        #     (y_pos_next > y_pos_last) * 3 + (y_pos_next < y_pos_last) * 4
        ### no move: 0
        ### right: x_pos > x_pos_old —> *1,
        ### left: x_pos < x_pos_old —> *2, 
        ### down: y_pos > y_pos_old —> *3, 
        ### up: y_pos < y_pos_old —> *4.

        # remove last step in sequence
        # test_data = test_data[:,:,:-1]

        # agent_label_v = np.array([labels_dict_agent_v[agent_v]] * len(remains))
        # agent_label_goal = np.array([labels_dict_agent_goal[agent_goal]] * len(remains))
        # env_label = np.array([labels_dict_env[env_N]] * len(remains))

        # datasets.append(VideoDataset(torch.tensor(test_data,dtype=torch.float32,device=device), 
        #                               torch.tensor(truth_data,dtype=torch.float32,device=device), 
        #                             #   torch.tensor(agent_poss,dtype=torch.float32,device=device), 
        #                               torch.tensor(truth_actions,dtype=torch.uint8, device=device),
        #                               torch.tensor(agent_label_v,device=device),
        #                               torch.tensor(agent_label_goal,device=device),
        #                               torch.tensor(env_label,device=device),
        #                               torch.tensor(target_srs,device=device))
        # )
        
    dataset = ConcatDataset(datasets)
    val_size = int(args.val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    print('val ids: ', dataset_val.indices, '\n')
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=0, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=len(dataset_val), num_workers=0, shuffle=True)
    # dataset_train = dataset
    # dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=0, shuffle=True)

    ## load trained VAE model
    model_VAE = RLNetwork(args.seq_length, 3, 16, 4, 16)
    if args.best_model:
        checkpoint = torch.load(model_dir + 'best_model.pt', map_location=torch.device(device))
    else:
        checkpoint = torch.load(model_dir + 'last_model.pt', map_location=torch.device(device))
    model_VAE.load_state_dict(checkpoint['model_state_dict'])
    model_VAE.to(device)
    # model_VAE.eval()

    ## define downstream MLP model
    model_MLP = MLPNetwork(in_dims=200, hidden_dims=128,
                           out_dims_goal=3, out_dims_v=3,
                           out_dims_env=3, out_dims_action=5,
                           conv_channels_in=3, conv_channels_hidden=8)
    model_MLP.to(device)
    model_MLP.train()

    optimizer = optim.Adam(model_MLP.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.scheduler_gamma)
    
    criterion = F.cross_entropy
    criterion_val = nn.CrossEntropyLoss(reduction='sum')

    max_acc_v = 0.0
    max_acc_goal = 0.0
    max_acc_env = 0.0
    max_acc_action = 0.0
    min_loss = 1e4
    min_loss_val = 1e4

    logs_sr = {}
    logs_prediction = {}
    logs_truth = {}
    logs_sr_val = {}
    logs_prediction_val = {}
    logs_truth_val = {}

    for epoch in range(args.epochs):
        save_flag = False
        logging = False

        loss_v = 0.0
        loss_goal = 0.0
        loss_env = 0.0
        loss_action = 0.0
        loss_sr = 0.0

        acc_v = 0.0
        acc_goal = 0.0
        acc_env = 0.0
        acc_action = 0.0

        if (epoch+1) % args.log_interval == 0:
            logging = True
            counter_log = 0
        # log_classes_v = []
        # log_classes_goal = []
        # log_classes_env = []
        # log_classes_action = []
        log_truth = []
        log_prediction = []
        log_sr = []

        for batch_id, batch_data in enumerate(dataloader_train):
            optimizer.zero_grad()
            
            seqs_train, grids_truth, truth_actions, agent_v_class, agent_goal_class, env_class, target_sr = batch_data
            _, _, mu, _, projections = model_VAE(seqs_train)

            # ## calculate truth actions
            # # agent pos at next step
            # agent_pos_next = (seqs_train[:,:,-1] - grids_truth)[:,0].flatten(1).argmax(1)
            # y_pos_next = torch.div(agent_pos_next, 22, rounding_mode='trunc')
            # x_pos_next = torch.remainder(agent_pos_next, 22)
            # # agent pos at last step in seq
            # agent_pos_last = (seqs_train[:,:,-2] - grids_truth)[:,0].flatten(1).argmax(1)
            # y_pos_last = torch.div(agent_pos_last, 22, rounding_mode='trunc')
            # x_pos_last = torch.remainder(agent_pos_last, 22)
            # # action based on pos change
            # truth_actions = (x_pos_next > x_pos_last) * 1 + (x_pos_next < x_pos_last) * 2 + \
            #     (y_pos_next > y_pos_last) * 3 + (y_pos_next < y_pos_last) * 4
            # ### x_pos > x_pos_old —> *1, 
            # ### x_pos < x_pos_old —> *2, 
            # ### y_pos > y_pos_old —> *3, 
            # ### y_pos < y_pos_old —> *4.
            
            ## train downstream model
            pred_goal, pred_v, pred_env, pred_action, pred_sr = model_MLP(projections)
            loss_goal = args.weight_loss_goal * criterion(pred_goal, agent_goal_class)
            loss_v = args.weight_loss_v * criterion(pred_v, agent_v_class)
            loss_env = args.weight_loss_env * criterion(pred_env, env_class)
            loss_action = args.weight_loss_action * criterion(pred_action, truth_actions)
            loss_sr = args.weight_loss_sr * ce_soft_label(target_sr, pred_sr) / len(target_sr)

            preds_goal_disc = pred_goal.argmax(dim=1)
            preds_v_disc = pred_v.argmax(dim=1)
            preds_env_disc = pred_env.argmax(dim=1)
            preds_action_disc = pred_action.argmax(dim=1)

            acc_goal += (preds_goal_disc==agent_goal_class).sum()
            acc_v += (preds_v_disc==agent_v_class).sum()
            acc_env += (preds_env_disc==env_class).sum()
            acc_action += (preds_action_disc==truth_actions).sum()

            loss = loss_goal + loss_v + loss_env + loss_action + loss_sr
            loss.backward()
            optimizer.step()

            if logging and counter_log < args.log_count:
                # log_classes_goal.append(agent_goal_class[:args.log_count-counter_log].detach())
                # log_classes_env.append(preds_env_disc[:args.log_count-counter_log].detach())
                # log_classes_v.append(agent_v_class[:args.log_count-counter_log].detach())
                # log_classes_action.append(truth_actions[:args.log_count-counter_log].detach())
                log_truth.append((agent_goal_class[:args.log_count-counter_log], 
                                  agent_v_class[:args.log_count-counter_log], 
                                  env_class[:args.log_count-counter_log], 
                                  truth_actions[:args.log_count-counter_log]
                                  ))
                log_prediction.append((preds_goal_disc[:args.log_count-counter_log], 
                                       preds_v_disc[:args.log_count-counter_log], 
                                       preds_env_disc[:args.log_count-counter_log], 
                                       preds_action_disc[:args.log_count-counter_log]
                                       ))
                log_sr.append((target_sr[:args.log_count-counter_log].detach().cpu().numpy(), 
                               pred_sr[:args.log_count-counter_log].detach().cpu().numpy()))
                counter_log += min(len(pred_sr), args.log_count-counter_log)

        # loss_goal /= len(dataset)
        # loss_v /= len(dataset)
        # loss_env /= len(dataset)
        # loss_action /= len(dataset)

        # loss =  loss_goal + loss_v + loss_env + loss_action

        acc_goal = acc_goal.float() / len(dataset)
        acc_v = acc_v.float() / len(dataset)
        acc_env = acc_env.float() / len(dataset)
        acc_action = acc_action.float() / len(dataset)

        ## update
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        scheduler.step()

        ## log numbers to tensorboard
        print('epoch {} acc_goal {:.4f} acc_v {:.4f} acc_env {:.4f} acc_action {:.4f}\n\
                        loss_goal {:.4f} loss_v {:.4f} loss_env {:.4f} loss_action {:.4f} loss_sr {:.4f}\n\
                        loss {:.4f} lr {:.5f}'.\
              format(epoch, acc_goal.item(), acc_v.item(), acc_env.item(), acc_action.item(), \
                     loss_goal.item(), loss_v.item(), loss_env.item(), loss_action.item(), loss_sr.item(),\
                     loss.item(), scheduler.get_last_lr()[0]))
        writer.add_scalar('loss/train', loss.item(), epoch)
        writer.add_scalar('loss_goal/train', loss_goal.item(), epoch)
        writer.add_scalar('loss_v/train', loss_v.item(), epoch)
        writer.add_scalar('loss_env/train', loss_env.item(), epoch)
        writer.add_scalar('loss_action/train', loss_action.item(), epoch)
        writer.add_scalar('loss_sr/train', loss_sr.item(), epoch)
        writer.add_scalar('acc_goal/train', acc_goal.item(), epoch)
        writer.add_scalar('acc_v/train', acc_v.item(), epoch)
        writer.add_scalar('acc_env/train', acc_env.item(), epoch)
        writer.add_scalar('acc_action/train', acc_action.item(), epoch)
        writer.add_scalar('lr/train', scheduler.get_last_lr()[0], epoch)

        if (epoch+1) % args.log_interval == 0:
            ## log sr to board
            writer.add_images('pred_sr_0.5/train', pred_sr[:,0][:args.log_count,None,...], epoch, dataformats='NCWH')
            writer.add_images('pred_sr_0.9/train', pred_sr[:,1][:args.log_count,None,...], epoch, dataformats='NCWH')
            writer.add_images('pred_sr_0.99/train', pred_sr[:,2][:args.log_count,None,...], epoch, dataformats='NCWH')
            writer.add_images('truth_sr_0.5/train', target_sr[:,0][:args.log_count,None,...], epoch, dataformats='NCWH')
            writer.add_images('truth_sr_0.9/train', target_sr[:,1][:args.log_count,None,...], epoch, dataformats='NCWH')
            writer.add_images('truth_sr_0.99/train', target_sr[:,2][:args.log_count,None,...], epoch, dataformats='NCWH')
            
            ## log sr to file
            logs_sr[epoch] = log_sr
            logs_prediction[epoch] = log_prediction
            logs_truth[epoch] = log_truth

            ## validation
            model_MLP.eval()

            loss_v_val = 0.0
            loss_goal_val = 0.0
            loss_env_val = 0.0
            loss_action_val = 0.0
            loss_sr_val = 0.0

            acc_v_val = 0.0
            acc_goal_val = 0.0
            acc_env_val = 0.0
            acc_action_val = 0.0

            log_truth = []
            log_prediction = []
            log_sr = []
            counter_log = 0

            with torch.no_grad():
                for batch_id, batch_data in enumerate(dataloader_val):
                    seqs_val, grids_truth, truth_actions, agent_v_class, agent_goal_class, env_class, target_sr = batch_data
                    _, _, mu, _, projections = model_VAE(seqs_val)

                    pred_goal, pred_v, pred_env, pred_action, pred_sr = model_MLP(projections)
                    loss_goal_val += args.weight_loss_goal * criterion_val(pred_goal, agent_goal_class)
                    loss_v_val += args.weight_loss_v * criterion_val(pred_v, agent_v_class)
                    loss_env_val += args.weight_loss_env * criterion_val(pred_env, env_class)
                    loss_action_val += args.weight_loss_action * criterion_val(pred_action, truth_actions)
                    loss_sr_val += args.weight_loss_sr * ce_soft_label(target_sr, pred_sr) / len(target_sr)

                    preds_goal_disc = pred_goal.argmax(dim=1)
                    preds_v_disc = pred_v.argmax(dim=1)
                    preds_env_disc = pred_env.argmax(dim=1)
                    preds_action_disc = pred_action.argmax(dim=1)

                    acc_goal_val += (preds_goal_disc==agent_goal_class).sum()
                    acc_v_val += (preds_v_disc==agent_v_class).sum()
                    acc_env_val += (preds_env_disc==env_class).sum()
                    acc_action_val += (preds_action_disc==truth_actions).sum()

                    if counter_log < args.log_count:
                        log_truth.append((agent_goal_class[:args.log_count-counter_log], 
                                        agent_v_class[:args.log_count-counter_log], 
                                        env_class[:args.log_count-counter_log], 
                                        truth_actions[:args.log_count-counter_log]
                                        ))
                        log_prediction.append((preds_goal_disc[:args.log_count-counter_log], 
                                            preds_v_disc[:args.log_count-counter_log], 
                                            preds_env_disc[:args.log_count-counter_log], 
                                            preds_action_disc[:args.log_count-counter_log]
                                            ))
                        log_sr.append((target_sr[:args.log_count-counter_log].detach().cpu().numpy(), 
                                       pred_sr[:args.log_count-counter_log].detach().cpu().numpy()))
                        counter_log += min(len(pred_sr), args.log_count-counter_log)

                
                loss_goal_val /= len(dataset_val)
                loss_v_val /= len(dataset_val)
                loss_env_val /= len(dataset_val)
                loss_action_val /= len(dataset_val)
                loss_sr_val /= len(dataset_val)

                loss_val =  loss_goal_val + loss_v_val + loss_env_val + loss_action_val + loss_sr_val
                
                acc_goal_val = acc_goal_val.float() / len(dataset_val)
                acc_v_val = acc_v_val.float() / len(dataset_val)
                acc_env_val = acc_env_val.float() / len(dataset_val)
                acc_action_val = acc_action_val.float() / len(dataset_val)

                logs_sr_val[epoch] = log_sr
                logs_prediction_val[epoch] = log_prediction
                logs_truth_val[epoch] = log_truth

                print('eval: acc_goal {:.4f} acc_v {:.4f} acc_env {:.4f} acc_action {:.4f}\n\
                                loss_goal {:.4f} loss_v {:.4f} loss_env {:.4f} loss_action {:.4f} loss_sr {:.4f}\n\
                                loss {:.4f}'.\
                    format(acc_goal_val.item(), acc_v_val.item(), acc_env_val.item(), acc_action_val.item(), \
                            loss_goal_val.item(), loss_v_val.item(), loss_env_val.item(), loss_action_val.item(), loss_sr_val.item(),\
                            loss_val.item()))
                writer.add_scalar('loss/val', loss_val.item(), epoch)
                writer.add_scalar('loss_goal/val', loss_goal_val.item(), epoch)
                writer.add_scalar('loss_v/val', loss_v_val.item(), epoch)
                writer.add_scalar('loss_env/val', loss_env_val.item(), epoch)
                writer.add_scalar('loss_action/val', loss_action_val.item(), epoch)
                writer.add_scalar('loss_sr/val', loss_sr_val.item(), epoch)
                writer.add_scalar('acc_goal/val', acc_goal_val.item(), epoch)
                writer.add_scalar('acc_v/val', acc_v_val.item(), epoch)
                writer.add_scalar('acc_env/val', acc_env_val.item(), epoch)
                writer.add_scalar('acc_action/val', acc_action_val.item(), epoch)

                writer.add_images('pred_sr_0.5/val', pred_sr[:,0][:args.log_count,None,...], epoch, dataformats='NCWH')
                writer.add_images('pred_sr_0.9/val', pred_sr[:,1][:args.log_count,None,...], epoch, dataformats='NCWH')
                writer.add_images('pred_sr_0.99/val', pred_sr[:,2][:args.log_count,None,...], epoch, dataformats='NCWH')
                writer.add_images('truth_sr_0.5/val', target_sr[:,0][:args.log_count,None,...], epoch, dataformats='NCWH')
                writer.add_images('truth_sr_0.9/val', target_sr[:,1][:args.log_count,None,...], epoch, dataformats='NCWH')
                writer.add_images('truth_sr_0.99/val', target_sr[:,2][:args.log_count,None,...], epoch, dataformats='NCWH')
        
            if loss_val <= min_loss_val and args.save_model:
                min_loss_val = loss_val
                torch.save({'epoch': epoch,
                        'model_state_dict': model_MLP.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                }, save_dir+'best_model.pt')
                save_flag = True

            model_MLP.train()

        ## save model checkpoint: best
        if loss <= min_loss and args.save_model and not save_flag:
        # if loss <= min_loss:
            min_loss = loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model_MLP.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
            }, save_dir+'best_model.pt')

        ## log embeddings to tensorboard
        # if (epoch+1) % args.log_interval == 0:
        #     writer.add_embedding(torch.concat(log_embeds_enc), metadata=torch.concat(log_classes_agent).tolist(), global_step=epoch, tag='enc_embeds_agent_test')
        #     writer.add_embedding(torch.concat(log_embeds_enc), metadata=torch.concat(log_classes_env).tolist(), global_step=epoch, tag='enc_embeds_env_test')
    
    ## save model checkpoint: last
    if args.save_model:
        torch.save({'epoch': epoch,
                    'model_state_dict': model_MLP.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
        }, save_dir+'last_model.pt')

    ## save logs to files
    pickle.dump(logs_sr, open(save_dir+'sr.pkl', 'wb'))   # target SRs and predicted SRs: train
    pickle.dump(logs_prediction, open(save_dir+'predictions.pkl', 'wb'))   # predicted goal, v, env, action: train
    pickle.dump(logs_truth, open(save_dir+'truth.pkl', 'wb'))   # target SRs and predicted SRs: eval
    pickle.dump(logs_sr_val, open(save_dir+'sr-test.pkl', 'wb'))   # target SRs and predicted SRs: eval
    pickle.dump(logs_prediction_val, open(save_dir+'predictions-test.pkl', 'wb'))   # target SRs and predicted SRs: eval
    pickle.dump(logs_truth_val, open(save_dir+'truth-test.pkl', 'wb'))   # target SRs and predicted SRs: eval

    writer.close()
