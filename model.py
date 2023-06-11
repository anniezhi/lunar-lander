import torch
import torch.nn as nn
import torch.nn.functional as F
from more_itertools import locate

# class Encoder3d(nn.Module):
#     def __init__(self, rows, cols, in_channels):
#         super(Encoder3d, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(rows * cols * in_channels * 20, 256)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(256, 256)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(256, 200)
#         # self.relu3 = nn.ReLU()
#         # self.fc4 = nn.Linear(256, 200)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.fc3(x)
#         # x = self.relu3(x)
#         # x = self.fc4(x)
#         return x

class Encoder3d(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # [bs, 16 (C), 20 (T), 128 (H), 128 (W)]
        # [bs, 16 (C), 10 (T), 64 (H), 64 (W)]
        x = self.pool(F.relu(self.conv2(x)))
        # [bs, 16 (C), 10 (T), 64 (H), 64 (W)]
        # [bs, 16 (C), 5 (T), 32 (H), 32 (W)]
        x = self.pool(F.relu(self.conv3(x)))
        # [bs, 4 (C), 5 (T), 32 (H), 32 (W)]
        # [bs, 4 (C), 2 (T), 16 (H), 16 (W)]
        return x
        
class Decoder2d(nn.Module):
    def __init__(self, rows, in_dims, in_channels, hidden_channels, out_channels):
        super(Decoder2d, self).__init__()
        self.in_channels = in_channels
        self.rows = rows
        # self.head = nn.Linear(in_dims, in_channels*(32**2))
        self.head = nn.Linear(in_dims, in_channels*(int(rows/4)**2))
        self.convt0 = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.convt1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2)
        self.convt2 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.head(x)
        # x = x.reshape(x.shape[0],self.in_channels, 32, 32)
        x = x.reshape(x.shape[0],self.in_channels, int(self.rows/4), int(self.rows/4))
        out = F.relu(self.convt0(x))
        out = F.relu(self.convt1(out))
        out = torch.sigmoid(self.convt2(out))
        return out

class RLNetwork(nn.Module):
    def __init__(self, rows, cols, in_channels, hidden_channels_enc, out_channels_enc, hidden_channels_dec) -> None:
        super(RLNetwork, self).__init__()
        # self.encoder = Encoder3d(rows, cols, in_channels)
        self.encoder = Encoder3d(in_channels, hidden_channels_enc, out_channels_enc)
        
        # self.head_mu = nn.Linear(2048, 200)
        self.head_mu = nn.Linear(int(rows*cols/8), 200)
        # self.head_logvar = nn.Linear(2048, 200)
        self.head_logvar = nn.Linear(int(rows*cols/8), 200)

        self.decoder_e = Decoder2d(rows, 200, out_channels_enc, hidden_channels_dec, in_channels)
        self.decoder_a = Decoder2d(rows, 200, out_channels_enc, hidden_channels_dec, in_channels)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        return mu

    def forward(self, data):
        x = self.encoder(data)
        # mu = self.head_mu(x)
        # logvar = self.head_logvar(x)

        x_flat = x.flatten(1)
        mu = self.head_mu(x_flat)
        logvar = self.head_logvar(x_flat)

        z = self.reparameterize(mu, logvar)

        recon_grid = self.decoder_e(z)
        recon_agent = self.decoder_a(z)
        return recon_grid, recon_agent, mu, logvar, z
    

class NCELoss(nn.Module):
    def __init__(self, temp) -> None:
        super(NCELoss, self).__init__()
        self.temp = temp

    # def loss_1(self, projs):
    #     z = F.normalize(projs, p=2, dim=1)
    #     sims = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    #     mask = (~torch.eye(len(z),len(z),dtype=bool)).float()
    #     nominator = torch.exp(sims / self.temp) * mask
    #     return -torch.log(torch.sum(nominator) / 2)

    # def loss_2(self, projs, labels, unique_labels):
    #     index_1 = list(locate(labels, lambda x: x == unique_labels[0]))
    #     index_2 = list(locate(labels, lambda x: x == unique_labels[1]))
    #     projs_1 = projs[index_1]
    #     projs_2 = projs[index_2]

    #     z_1 = F.normalize(projs_1, p=2, dim=1)
    #     z_2 = F.normalize(projs_2, p=2, dim=1)
    #     reps = torch.cat([z_1, z_2], dim=0)
    #     mask_diag = (~torch.eye(len(projs),len(projs),dtype=bool)).float()
    #     mask_cat = torch.block_diag(torch.ones(len(z_1), len(z_1)), torch.ones(len(z_2), len(z_2)))

    #     sim_mat = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2)
    #     sim_mat_exp = torch.exp(sim_mat / self.temp)

    #     nom = torch.sum(sim_mat_exp * mask_cat * mask_diag, dim=1) / torch.sum(mask_cat * mask_diag, dim=1)
    #     denom = torch.sum(sim_mat_exp * mask_diag, dim=1)

    #     all_losses = -torch.log(nom / denom)
    #     loss = torch.mean(all_losses)
    #     return loss

    def loss_multi(self, projs, labels, unique_labels):
        z = []
        for l in unique_labels:
            index_l = list(locate(labels, lambda x: x == l))
            projs_l = projs[index_l]
            z.append(F.normalize(projs_l, p=2, dim=1))
        reps = torch.cat(z, dim=0)
        mask_diag = (~torch.eye(len(projs),len(projs),dtype=bool)).float().to(reps.device)
        mask_cat = torch.block_diag(*[torch.ones(len(zi),len(zi)) for zi in z]).to(reps.device)
        
        sim_mat = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2)
        sim_mat_exp = torch.exp(sim_mat / self.temp)

        nom = torch.zeros(len(projs)).to(reps.device)
        mask_nom = (torch.sum(mask_cat * mask_diag, dim=1) != 0)
        nom[mask_nom] = torch.sum(sim_mat_exp * mask_cat * mask_diag, dim=1)[mask_nom] / torch.sum(mask_cat * mask_diag, dim=1)[mask_nom]
        denom = torch.sum(sim_mat_exp * mask_diag, dim=1)

        all_losses = torch.zeros(len(projs)).to(reps.device)
        all_losses[mask_nom] = -torch.log(nom[mask_nom] / denom[mask_nom])
        loss = torch.mean(all_losses)
        return loss

    def forward(self, projs, labels):
        unique_labels = list(set(labels))
        
        ## only positive case
        # if len(unique_labels) == 1:
        #     return self.loss_1(projs)
        # elif len(unique_labels) == 2:
        #     return self.loss_2(projs, labels, unique_labels)
        # else:
        return self.loss_multi(projs, labels, unique_labels)
    
class MLPNetwork(nn.Module):
    def __init__(self, in_dims, out_dims_agent_type, out_dims_action, 
                 hidden_dims, conv_channels_in, conv_channels_hidden, 
                 num_discount=3, image_size=64) -> None:
        super(MLPNetwork, self).__init__()
        self.linear_1 = nn.Linear(in_dims, hidden_dims)
        self.linear_2 = nn.Linear(hidden_dims, hidden_dims)
        self.head_agent_type = nn.Linear(hidden_dims, out_dims_agent_type)
        self.head_action = nn.Linear(hidden_dims, out_dims_action)
        self.head_sr_1 = nn.Linear(hidden_dims, conv_channels_in*image_size*image_size)
        self.conv_channels_in = conv_channels_in
        self.image_size = image_size
        self.head_sr_2 = nn.Sequential(
            nn.Conv2d(conv_channels_in, conv_channels_hidden, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels_hidden, num_discount, kernel_size=1, stride=1),
            # nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        pred_agent_type = F.softmax(self.head_agent_type(x), dim=1)
        pred_action = F.softmax(self.head_action(x), dim=1)
        pred_sr = F.relu(self.head_sr_1(x)).view(-1, self.conv_channels_in, self.image_size, self.image_size)
        pred_sr = self.head_sr_2(pred_sr)
        pred_sr = F.softmax(pred_sr.reshape(pred_sr.size(0), pred_sr.size(1), -1), 2).view_as(pred_sr)
        return pred_agent_type, pred_action, pred_sr