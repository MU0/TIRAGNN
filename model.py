from torch import nn
import torch.nn.functional as F
import torch

torch.backends.cudnn.benchmark = True
from SelfAttentionLSTM import SelfAttentionLSTM


Temporal_layer = 4


class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


class _Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_Aggregation, self).__init__()
        self.aggre = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.aggre(x)


class _UserModel(nn.Module):
    ''' User modeling to learn user latent factors.
    User modeling leverages two types aggregation: interest aggregation and social aggregation
    '''

    def __init__(self, emb_dim, user_emb, item_emb, rate_emb, batch_size, device):
        super(_UserModel, self).__init__()
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.device = (torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}'))

        self.w1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w4 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w5 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w6 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w7 = nn.Linear(self.emb_dim, self.emb_dim)

        self.g_v = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        # ADD GSF
        self.g_sf = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)

        self.user_items_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items = _Aggregation(self.emb_dim, self.emb_dim)

        self.user_items_att_s1 = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items_s1 = _Aggregation(self.emb_dim, self.emb_dim)

        self.user_users_att_s2 = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_neigbors_s2 = _Aggregation(self.emb_dim, self.emb_dim)

        self.u_user_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.u_aggre_neigbors = _Aggregation(self.emb_dim, self.emb_dim)

        # add
        self.sf_user_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.sf_aggre_neigbors = _Aggregation(self.emb_dim, self.emb_dim)

        self.user_items_att_sf1 = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items_sf1 = _Aggregation(self.emb_dim, self.emb_dim)
        self.sf_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_sf_neigbors = _Aggregation(self.emb_dim, self.emb_dim)

        #    self.temporal = LSTMModel(self.emb_dim, self.emb_dim, Temporal_layer,0.5,0.5)
        self.temporal = SelfAttentionLSTM(self.emb_dim, self.emb_dim, Temporal_layer, self.emb_dim)

        self.h0 = torch.randn(Temporal_layer, self.batch_size, self.emb_dim).to(self.device)
        self.c0 = torch.randn(Temporal_layer, self.batch_size, self.emb_dim).to(self.device)
        '''
        self.combine_mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.emb_dim, 2*self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2*self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU()
        )    '''
        self.combine_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(5 * self.emb_dim, 3 * self.emb_dim, bias=True),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(3 * self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),

        )
        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10

    def forward(self, uids, u_item_pad, u_user_pad, u_user_item_pad, sf_user_pad, sf_user_item_pad):

        ## item embedding 项目兴趣聚合
        q_j = self.item_emb(u_item_pad[:, :, 0])  # B x maxi_len x emb_dim
        mask_u = torch.where(u_item_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))  # B x maxi_len
        e_ij = self.rate_emb(u_item_pad[:, :, 1])  # B x maxi_len x emb_dim
        x_ij = self.g_v(torch.cat([q_j, e_ij], dim=2).view(-1, 2 * self.emb_dim)).view(
            q_j.size())  # B x maxi_len x emb_dim
        p_i = mask_u.unsqueeze(2).expand_as(q_j) * self.user_emb(uids).unsqueeze(1).expand_as(
            q_j)  # B x maxi_len x emb_dim

        ## interest aggregation
        # long-term static interest
        alpha = self.user_items_att(torch.cat([self.w1(x_ij), self.w1(p_i)], dim=2).view(-1, 2 * self.emb_dim)).view(
            mask_u.size())  # B x maxi_len
        alpha = torch.exp(alpha) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)

        h_iL = self.aggre_items(torch.sum(alpha.unsqueeze(2).expand_as(x_ij) * x_ij, 1))  # B x emb_dim

        # short-term dynamic interest

        temporal_ia, (h_n, c_n) = self.temporal(x_ij, (self.h0, self.c0))
        #  temporal_ia = self.temporal1(x_ij)
        h_iS = temporal_ia[:, -1, :]

        h_iI = h_iL * h_iS
        h_iI = F.dropout(h_iI, 0.5, training=self.training)

        ## social aggregation   explict user 显示邻居项目聚合
        q_j_s = self.item_emb(u_user_item_pad[:, :, :, 0])  # B x maxu_len x maxi_len x emb_dim
        mask_s = torch.where(u_user_item_pad[:, :, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))  # B x maxu_len x maxi_len
        p_i_s = mask_s.unsqueeze(3).expand_as(q_j_s) * self.user_emb(u_user_pad).unsqueeze(2).expand_as(
            q_j_s)  # B x maxu_len x maxi_len x emb_dim
        u_user_item_er = self.rate_emb(u_user_item_pad[:, :, :, 1])  # B x maxu_len x maxi_len x emb_dim
        x_ij_s = self.g_v(torch.cat([q_j_s, u_user_item_er], dim=3).view(-1, 2 * self.emb_dim)).view(
            q_j_s.size())  # B x maxu_len x maxi_len x emb_dim

        alpha_s = self.user_items_att_s1(
            torch.cat([self.w2(x_ij_s), self.w2(p_i_s)], dim=3).view(-1, 2 * self.emb_dim)).view(
            mask_s.size())  # B x maxu_len x maxi_len
        alpha_s = torch.exp(alpha_s) * mask_s
        alpha_s = alpha_s / (torch.sum(alpha_s, 2).unsqueeze(2).expand_as(alpha_s) + self.eps)

        h_oL_temp = torch.sum(alpha_s.unsqueeze(3).expand_as(x_ij_s) * x_ij_s, 2)  # B x maxu_len x emb_dim
        h_oL = self.aggre_items_s1(h_oL_temp.view(-1, self.emb_dim)).view(h_oL_temp.size())  # B x maxu_len x emb_dim

        h_oS = []
        for i in range(x_ij_s.shape[1]):
            output_s, (h_n, c_n) = self.temporal(x_ij_s[:, i, :, :], (self.h0, self.c0))
            h_oS.append(output_s[:, -1, :])

        h_oS = torch.stack(h_oS).permute(1, 0, 2)

        h_oI = h_oL * h_oS
        h_oI = F.dropout(h_oI, p=0.5, training=self.training)

        # soical influence
        mask_su = torch.where(u_user_pad > 0, torch.tensor([1.], device=self.device),
                              torch.tensor([0.], device=self.device))

        beta = self.user_users_att_s2(
            torch.cat([self.w3(h_oI), self.w3(self.user_emb(u_user_pad))], dim=2).view(-1, 2 * self.emb_dim)).view(
            u_user_pad.size())
        beta = torch.exp(beta) * mask_su
        beta = beta / (torch.sum(beta, 1).unsqueeze(1).expand_as(beta) + self.eps)
        h_iN = self.aggre_neigbors_s2(torch.sum(beta.unsqueeze(2).expand_as(h_oI) * h_oI, 1))  # B x emb_dim
        h_iN = F.dropout(h_iN, p=0.5, training=self.training)

        # sf_user implict user 隐式邻居项目聚合
        q_a_f = self.item_emb(sf_user_item_pad[:, :, :, 0])  # B x maxu_len x maxi_len x emb_dim
        mask_sf = torch.where(sf_user_item_pad[:, :, :, 0] > 0, torch.tensor([1.], device=self.device),
                              torch.tensor([0.], device=self.device))  # B x maxu_len x maxi_len
        sf_user_item_er = self.rate_emb(sf_user_item_pad[:, :, :, 1])  # B x maxu_len x maxi_len x emb_dim
        x_ia_sf = self.g_v(torch.cat([q_a_f, sf_user_item_er], dim=3).view(-1, 2 * self.emb_dim)).view(
            q_a_f.size())  # B x maxu_len x maxi_len x emb_dim

        p_i_sf = mask_sf.unsqueeze(3).expand_as(q_a_f) * self.user_emb(sf_user_pad[:, :, 0]).unsqueeze(2).expand_as(
            q_a_f)  # B x maxu_len x maxi_len x emb_dim

        alpha_i_sf = self.user_items_att_sf1(
            torch.cat([self.w2(x_ia_sf), self.w2(p_i_sf)], dim=3).view(-1, 2 * self.emb_dim)).view(
            mask_sf.size())  # B x maxu_len x maxi_len
        alpha_i_sf = torch.exp(alpha_i_sf) * mask_sf
        alpha_i_sf = alpha_i_sf / (torch.sum(alpha_i_sf, 2).unsqueeze(2).expand_as(alpha_i_sf) + self.eps)

        h_sfI_temp = torch.sum(alpha_i_sf.unsqueeze(3).expand_as(x_ia_sf) * self.w2(x_ia_sf),
                               2)  # B x maxu_len x emb_dim
        h_sfIs = self.aggre_items_sf1(h_sfI_temp.view(-1, self.emb_dim)).view(
            h_sfI_temp.size())  # B x maxu_len x emb_dim

        h_sfIL = []
        for i in range(p_i_sf.shape[1]):
            output_s, (h_n, c_n) = self.temporal(p_i_sf[:, i, :, :], (self.h0, self.c0))
            h_sfIL.append(output_s[:, -1, :])

        h_sfIL = torch.stack(h_sfIL).permute(1, 0, 2)

        h_sfI = h_sfIL * h_sfIs
        h_sfI = F.dropout(h_sfI, 0.5, training=self.training)

        mask_u_f = torch.where(sf_user_pad[:, :, 0] > 0, torch.tensor([1.0], device=self.device),
                               torch.tensor([0.], device=self.device))

        ##calculate attention score
        p_u_sf = mask_u_f.unsqueeze(2).expand_as(h_sfI) * self.user_emb(uids).unsqueeze(1).expand_as(h_sfI)
        # p_u_sf = self.user_emb(sf_user_pad[:,:,0])
        beta_sf = self.sf_users_att(
            torch.cat([self.w3(h_sfI), self.w3(p_u_sf)], dim=2).view(-1, 2 * self.emb_dim)).view(mask_u_f.size())
        beta_sf = torch.exp(beta_sf) * mask_u_f
        beta_sf = beta_sf / (torch.sum(beta_sf, 1).unsqueeze(1).expand_as(beta_sf) + self.eps)

        h_i_sf = self.aggre_sf_neigbors(torch.sum(beta_sf.unsqueeze(2).expand_as(h_sfI) * self.w3(h_sfI), 1))
        h_i_sf = F.dropout(h_i_sf, p=0.5, training=self.training)

        # user aggreation  显示邻居聚合
        su = self.user_emb(u_user_pad)
        mask_su = torch.where(u_user_pad > 0, torch.tensor([1.], device=self.device),
                              torch.tensor([0.], device=self.device))
        p_uf = mask_su.unsqueeze(2).expand_as(su) * self.user_emb(uids).unsqueeze(1).expand_as(su)
        alpha_su = self.u_user_users_att(torch.cat([self.w6(su), self.w6(p_uf)], dim=2)).view(mask_su.size())
        # alpha_su = torch.matmul(su, F.tanh(self.user_users_att(ti_emb)).unsqueeze(2)).squeeze()
        alpha_su = torch.exp(alpha_su) * mask_su
        alpha_su = alpha_su / (torch.sum(alpha_su, 1).unsqueeze(1).expand_as(alpha_su) + self.eps)

        h_su = self.u_aggre_neigbors(torch.sum(alpha_su.unsqueeze(2).expand_as(su) * self.w6(su), 1))
        h_su = F.dropout(h_su, p=0.5, training=self.training)

        # 隐式邻居聚合
        sf = self.user_emb(sf_user_pad[:, :, 0])
        p_sf = mask_u_f.unsqueeze(2).expand_as(sf) * self.user_emb(uids).unsqueeze(1).expand_as(sf)
        alpha_sf = self.sf_user_users_att(torch.cat([self.w7(sf), self.w7(p_sf)], dim=2)).view(mask_u_f.size())
        alpha_sf = torch.exp(alpha_sf) * mask_u_f
        alpha_sf = alpha_sf / (torch.sum(alpha_sf, 1).unsqueeze(1).expand_as(alpha_sf) + self.eps)

        h_sf = self.sf_aggre_neigbors(torch.sum(alpha_sf.unsqueeze(2).expand_as(sf) * self.w7(sf), 1))
        h_sf = F.dropout(h_sf, p=0.5, training=self.training)

        ## user latent factor
        h = self.combine_mlp(torch.cat([h_iI, h_iN, h_i_sf, h_sf, h_su], dim=1))

        return h


class _ItemModel(nn.Module):
    '''Item modeling to learn item latent factors.
    Item modeling leverages two types aggregation: attraction aggregation and correlative aggregation
    '''

    def __init__(self, emb_dim, user_emb, item_emb, rate_emb, batch_size, device):
        super(_ItemModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.batch_size = batch_size
        self.device = (torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}'))

        self.w1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w4 = nn.Linear(self.emb_dim, self.emb_dim)

        self.g_u = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.g_v = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)

        self.item_users_att_i = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_users_i = _Aggregation(self.emb_dim, self.emb_dim)

        self.item_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_users = _Aggregation(self.emb_dim, self.emb_dim)

        self.i_friends_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_i_friends = _Aggregation(self.emb_dim, self.emb_dim)

        self.if_friends_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_if_friends = _Aggregation(self.emb_dim, self.emb_dim)

        self.temporal = SelfAttentionLSTM(self.emb_dim, self.emb_dim, Temporal_layer, self.emb_dim)
        self.h0 = torch.randn(Temporal_layer, self.batch_size, self.emb_dim).to(self.device)
        self.c0 = torch.randn(Temporal_layer, self.batch_size, self.emb_dim).to(self.device)

        self.combine_mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3 * self.emb_dim, 2 * self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU()
        )

        # used for preventing zero div error when calculating softmax score
        self.eps = 1e-10

    def forward(self, iids, i_user_pad, i_item_pad, i_item_user_pad):
        ## item embedding
        p_i = self.user_emb(i_user_pad[:, :, 0])
        mask_i = torch.where(i_user_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        e_ij = self.rate_emb(i_user_pad[:, :, 1])
        y_ji = self.g_u(torch.cat([p_i, e_ij], dim=2).view(-1, 2 * self.emb_dim)).view(p_i.size())
        q_j = mask_i.unsqueeze(2).expand_as(p_i) * self.item_emb(iids).unsqueeze(1).expand_as(p_i)

        ## attraction aggregation
        # long-term stable attraction
        miu = self.item_users_att_i(torch.cat([self.w1(y_ji), self.w1(q_j)], dim=2).view(-1, 2 * self.emb_dim)).view(
            mask_i.size())
        miu = torch.exp(miu) * mask_i
        miu = miu / (torch.sum(miu, 1).unsqueeze(1).expand_as(miu) + self.eps)
        z_jL = self.aggre_users_i(torch.sum(miu.unsqueeze(2).expand_as(y_ji) * self.w1(y_ji), 1))

        # short-term dynamic attraction
        temporal_jt, (h_n, c_n) = self.temporal(y_ji, (self.h0, self.c0))
        z_jS = temporal_jt[:, -1, :]

        z_jA = z_jL * z_jS
        z_jA = F.dropout(z_jA, p=0.5, training=self.training)

        ## correlative aggregation
        p_i_s = self.user_emb(i_item_user_pad[:, :, :, 0])
        mask_s = torch.where(i_item_user_pad[:, :, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        q_j_s = mask_s.unsqueeze(3).expand_as(p_i_s) * self.item_emb(i_item_pad).unsqueeze(2).expand_as(p_i_s)
        i_item_user_er = self.rate_emb(i_item_user_pad[:, :, :, 1])
        y_ji_s = self.g_u(torch.cat([p_i_s, i_item_user_er], dim=3).view(-1, 2 * self.emb_dim)).view(p_i_s.size())

        miu_s = self.i_friends_att(torch.cat([self.w2(y_ji_s), self.w2(q_j_s)], dim=3).view(-1, 2 * self.emb_dim)).view(
            mask_s.size())
        miu_s = torch.exp(miu_s) * mask_s
        miu_s = miu_s / (torch.sum(miu_s, 2).unsqueeze(2).expand_as(miu_s) + self.eps)

        z_kL_temp = torch.sum(miu_s.unsqueeze(3).expand_as(y_ji_s) * y_ji_s, 2)
        z_kL = self.aggre_i_friends(z_kL_temp.view(-1, self.emb_dim)).view(z_kL_temp.size())

        z_kS = []
        for i in range(y_ji_s.shape[1]):
            output_s, (h_n, c_n) = self.temporal(y_ji_s[:, i, :, :], (self.h0, self.c0))
            z_kS.append(output_s[:, -1, :])
        z_kS = torch.stack(z_kS).permute(1, 0, 2)

        z_kA = z_kL * z_kS
        z_kA = F.dropout(z_kA, p=0.5, training=self.training)

        # correlative influence
        mask_si = torch.where(i_item_pad > 0, torch.tensor([1.], device=self.device),
                              torch.tensor([0.], device=self.device))

        kappa = self.if_friends_att(
            torch.cat([self.w3(z_kA), self.w3(self.item_emb(i_item_pad))], dim=2).view(-1, 2 * self.emb_dim)).view(
            i_item_pad.size())
        kappa = torch.exp(kappa) * mask_si
        kappa = kappa / (torch.sum(kappa, 1).unsqueeze(1).expand_as(kappa) + self.eps)

        z_jN = self.aggre_if_friends(torch.sum(kappa.unsqueeze(2).expand_as(z_kA) * z_kA, 1))
        z_jN = F.dropout(z_jN, p=0.5, training=self.training)

        # item aggregation
        q_a = self.item_emb(i_item_pad)  # B x maxi_len x emb_dim
        mask_u = torch.where(i_item_pad > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))  # B x maxi_len
        ## calculate attention scores in item aggregation
        # x_ia & p_i cat时候，保证pad位置不能被cat上
        p_i = mask_u.unsqueeze(2).expand_as(q_a) * self.item_emb(iids).unsqueeze(1).expand_as(
            q_a)  # B x maxi_len x emb_dim
        # alpha = self.user_items_att(torch.cat([x_ia, p_i], dim = 2)) 就够了，B, maxi_len,1
        # 计算attention的另一种方法，之前是pygat中增加大矩阵
        alpha = self.i_friends_att(torch.cat([self.w2(q_a), self.w2(p_i)], dim=2).view(-1, 2 * self.emb_dim)).view(
            mask_u.size())  # B x maxi_len
        alpha = torch.exp(alpha) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)

        z_if = self.aggre_i_friends(torch.sum(alpha.unsqueeze(2).expand_as(q_a) * self.w2(q_a), 1))  # B x emb_dim
        z_if = F.dropout(z_if, p=0.5, training=self.training)

        ## item latent factor
        z = self.combine_mlp(torch.cat([z_jA, z_jN, z_if], dim=1))

        return z


class TIRAGNN(nn.Module):
    '''
    Args:
        number_users: the number of users in the dataset.
        number_items: the number of items in the dataset.
        num_rate_levels: the number of rate levels in the dataset.
        emb_dim: the dimension of user and item embedding (default = 128).
        batch_size: the number of samples selected for training (default = 256).
        device: the gpu/cpu where the training takes place.
    '''

    def __init__(self, num_users, num_items, num_rate_levels, emb_dim, batch_size, device):
        super(TIRAGNN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_rate_levels = num_rate_levels
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.device = device
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim, padding_idx=0)
        self.rate_emb = nn.Embedding(self.num_rate_levels, self.emb_dim, padding_idx=0)

        self.user_model = _UserModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb, self.batch_size,
                                     self.device)

        self.item_model = _ItemModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb, self.batch_size,
                                     self.device)

        self.rate_pred = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3 * self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(self.emb_dim, 1)
        )

    def forward(self, uids, iids, u_item_pad, u_user_pad, u_user_item_pad, i_user_pad, sf_user_pad, sf_user_item_pad,
                i_item_pad, i_item_user_pad):
        '''
        Args:
            uids: the user id sequences.
            iids: the item id sequences.
            u_item_pad: the padded user-item graph.
            u_user_pad: the padded user-user graph.
            u_user_item_pad: the padded user-user-item graph.
            i_user_pad: the padded item-user graph.
            i_item_pad: the padded item-item graph.
            i_item_user_pad: the padded item-item-user graph.

        Shapes:
            uids: (B).
            iids: (B).
            u_item_pad: (B, ItemSeqMaxLen, 3).
            u_user_pad: (B, UserSocMaxLen).
            u_user_item_pad: (B, UserSocMaxLen, ItemSeqMaxLen, 3).
            i_user_pad: (B, UserSeqMaxLen, 3).
            i_item_pad: (B, ItemCorMaxLen).
            i_item_user_pad: (B, ItemCorMaxLen, UserSeqMaxLen, 3).

        Returns:
            the predicted rate scores of the user to the item.
        '''

        h = self.user_model(uids, u_item_pad, u_user_pad, u_user_item_pad, sf_user_pad, sf_user_item_pad)
        z = self.item_model(iids, i_user_pad, i_item_pad, i_item_user_pad)

        x = torch.cat([h, z, h * z], dim=1)  # 拼接 h, z 和内积结果

        r_ij = self.rate_pred(x)
        return r_ij





