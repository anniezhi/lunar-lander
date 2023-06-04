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
import numpy as np
from PIL import Image
from model import *
from data_utils import *

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def ce_soft_label(target, pred):
    return -(target * torch.log(pred)).sum()

def dict2mdtable(d, key='Name', val='Value'):
    rows = [f'| {key} | {val} |']
    rows += ['|--|--|']
    rows += [f'| {k} | {v} |' for k, v in d.items()]
    return "  \n".join(rows)

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
parser.add_argument("--start-sample", type=int, default=0,
                    help="frame to start sampling")
parser.add_argument("--sample-interleave", type=int, default=1,
                    help="interleave of sampling during taking frames from video")
parser.add_argument("--rows", type=int, default=64,
                    help="rows of resized image")
parser.add_argument("--cols", type=int, default=64,
                    help="cols of resized image")
parser.add_argument("--best-model", default=False, action='store_true',
                    help="use best vae model if True, otherwise use last model")
parser.add_argument("--save-model", default=False, action='store_true',
                    help="save model after training")
parser.add_argument("--downstream-model", type=str, default=None,
                    help="names of the downstream RLRL evaluation models")
parser.add_argument("--weight-loss-agent", type=float, default=1.0,
                    help="Weight of loss term agent v (default: 1.0)")
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

    model_dir = os.getcwd() + '/lunar-lander/saved/' + args.vae_model + '/'
    if not os.path.exists(model_dir):
        print(f'VAE model {model_dir} not found')
    save_dir = os.getcwd() + '/lunar-lander/saved_downstream/' + args.downstream_model + '/'
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    writer = SummaryWriter(log_dir=save_dir)
    args_dict = vars(args)
    writer.add_text("Hyperparameters", dict2mdtable(args_dict))

    ## Load data

    datasets = []
    ROWS, COLS = args.rows, args.cols

    labels_dict_agent_type = {"laggy": 0, "optimal": 1}
    writer.add_text("Labels: agent label", dict2mdtable(labels_dict_agent_type))

    gamma = np.array([0.5, 0.9, 0.99])

    if args.data_root_dir is None:
        data_root_dir = os.getcwd() + '/lunar-lander/data_videos/'
    else:
        data_root_dir = args.data_root_dir

    for model in args.agent_models:
        root_dir = data_root_dir + model + '/'
        grid_file = 'env_grids.npy'
        agent_poss_file = 'agent_poss.npz'
        actions_file = 'actions.npz'

        sample_ids = [i+args.first_sample for i in range(args.num_samples)]

        datasets.append(VideoDataset(sample_ids, grid_file, agent_poss_file, actions_file,
                                     root_dir, model, args.start_sample, args.sample_interleave,
                                     args.rows, args.cols,
                                     args.seq_length, labels_dict_agent_type[model]))

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
    model_VAE = RLNetwork(ROWS, COLS, 3, 16, 4, 16)
    if args.best_model:
        checkpoint = torch.load(model_dir + 'best_model.pt', map_location=torch.device(device))
    else:
        checkpoint = torch.load(model_dir + 'last_model.pt', map_location=torch.device(device))
    model_VAE.load_state_dict(checkpoint['model_state_dict'])
    model_VAE.to(device)
    # model_VAE.eval()

    ## define downstream MLP model
    model_MLP = MLPNetwork(in_dims=200, hidden_dims=128,
                           out_dims_agent_type=2, out_dims_action=4,
                           conv_channels_in=3, conv_channels_hidden=8)
    model_MLP.to(device)
    model_MLP.train()

    optimizer = optim.Adam(model_MLP.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.scheduler_gamma)
    
    criterion = F.cross_entropy
    criterion_val = nn.CrossEntropyLoss(reduction='sum')

    max_acc_agent_type = 0.0
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

        loss_agent = 0.0
        loss_action = 0.0
        loss_sr = 0.0

        acc_agent = 0.0
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
            
            seqs_train, grids_truth, truth_actions, agent_class, target_sr = batch_data
            _, _, mu, _, projections = model_VAE(seqs_train)
            
            ## train downstream model
            pred_agent, pred_action, pred_sr = model_MLP(projections)
            loss_agent = args.weight_loss_agent * criterion(pred_agent, agent_class)
            loss_action = args.weight_loss_action * criterion(pred_action, truth_actions)
            loss_sr = args.weight_loss_sr * ce_soft_label(target_sr, pred_sr) / len(target_sr)

            preds_agent_disc = pred_agent.argmax(dim=1)
            preds_action_disc = pred_action.argmax(dim=1)

            acc_agent += (preds_agent_disc==agent_class).sum()
            acc_action += (preds_action_disc==truth_actions).sum()

            loss = loss_agent + loss_action + loss_sr
            loss.backward()
            optimizer.step()

            if logging and counter_log < args.log_count:
                # log_classes_goal.append(agent_goal_class[:args.log_count-counter_log].detach())
                # log_classes_action.append(truth_actions[:args.log_count-counter_log].detach())
                log_truth.append((agent_class[:args.log_count-counter_log], 
                                  truth_actions[:args.log_count-counter_log]
                                  ))
                log_prediction.append((preds_agent_disc[:args.log_count-counter_log], 
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

        acc_agent = acc_agent.float() / len(dataset)
        acc_action = acc_action.float() / len(dataset)

        ## update
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        scheduler.step()

        ## log numbers to tensorboard
        print('epoch {} acc_agent {:.4f} acc_action {:.4f}\n\
              loss_agent {:.4f} loss_action {:.4f} loss_sr {:.4f}\n\
              loss {:.4f} lr {:.5f}'.\
              format(epoch, acc_agent.item(), acc_action.item(), \
                     loss_agent.item(), loss_action.item(), loss_sr.item(),\
                     loss.item(), scheduler.get_last_lr()[0]))
        writer.add_scalar('loss/train', loss.item(), epoch)
        writer.add_scalar('loss_agent/train', loss_agent.item(), epoch)
        writer.add_scalar('loss_action/train', loss_action.item(), epoch)
        writer.add_scalar('loss_sr/train', loss_sr.item(), epoch)
        writer.add_scalar('acc_agent/train', acc_agent.item(), epoch)
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

            loss_agent_val = 0.0
            loss_action_val = 0.0
            loss_sr_val = 0.0

            acc_agent_val = 0.0
            acc_action_val = 0.0

            log_truth = []
            log_prediction = []
            log_sr = []
            counter_log = 0

            with torch.no_grad():
                for batch_id, batch_data in enumerate(dataloader_val):
                    seqs_val, grids_truth, truth_actions, agent_class, target_sr = batch_data
                    _, _, mu, _, projections = model_VAE(seqs_val)

                    pred_agent, pred_action, pred_sr = model_MLP(projections)
                    loss_agent_val += args.weight_loss_agent * criterion(pred_agent, agent_class)
                    loss_action_val += args.weight_loss_action * criterion(pred_action, truth_actions)
                    loss_sr_val += args.weight_loss_sr * ce_soft_label(target_sr, pred_sr) / len(target_sr)

                    preds_agent_disc = pred_agent.argmax(dim=1)
                    preds_action_disc = pred_action.argmax(dim=1)

                    acc_agent_val += (preds_agent_disc==agent_class).sum()
                    acc_action_val += (preds_action_disc==truth_actions).sum()

                    if counter_log < args.log_count:
                        log_truth.append((agent_class[:args.log_count-counter_log], 
                                          truth_actions[:args.log_count-counter_log]
                                        ))
                        log_prediction.append((preds_agent_disc[:args.log_count-counter_log], 
                                               preds_action_disc[:args.log_count-counter_log]
                                            ))
                        log_sr.append((target_sr[:args.log_count-counter_log].detach().cpu().numpy(), 
                                       pred_sr[:args.log_count-counter_log].detach().cpu().numpy()))
                        counter_log += min(len(pred_sr), args.log_count-counter_log)

                
                loss_agent_val /= len(dataset_val)
                loss_action_val /= len(dataset_val)
                loss_sr_val /= len(dataset_val)

                loss_val =  loss_agent_val + loss_action_val + loss_sr_val
                
                acc_agent_val = acc_agent_val.float() / len(dataset_val)
                acc_action_val = acc_action_val.float() / len(dataset_val)

                logs_sr_val[epoch] = log_sr
                logs_prediction_val[epoch] = log_prediction
                logs_truth_val[epoch] = log_truth

                print('eval: acc_agent {:.4f} acc_action {:.4f}\n\
                      loss_agent {:.4f} loss_action {:.4f} loss_sr {:.4f}\n\
                      loss {:.4f}'.\
                    format(acc_agent_val.item(), acc_action_val.item(), \
                            loss_agent_val.item(), loss_action_val.item(), loss_sr_val.item(),\
                            loss_val.item()))
                writer.add_scalar('loss/val', loss_val.item(), epoch)
                writer.add_scalar('loss_agent/val', loss_agent_val.item(), epoch)
                writer.add_scalar('loss_action/val', loss_action_val.item(), epoch)
                writer.add_scalar('loss_sr/val', loss_sr_val.item(), epoch)
                writer.add_scalar('acc_agent/val', acc_agent_val.item(), epoch)
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
