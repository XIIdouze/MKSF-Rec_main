import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import Attention


class LearnerKnowledge_Aggregator(nn.Module):

    def __init__(self, v_to_e, u_to_e, history_uv_lists, embed_dim, cuda="cpu", uv=True):
        super(LearnerKnowledge_Aggregator, self).__init__()
        self.uv = uv
        self.v_to_e = v_to_e
        self.u_to_e = u_to_e
        self.history_uv_lists = history_uv_lists
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)
        self.device = cuda

    def forward(self, nodes):
        history_uv = []
        for node in nodes:
            history_uv.append(self.history_uv_lists[int(node)])
        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)
        
        for i in range(len(history_uv)):
            history = history_uv[i]
            num_interact = len(history)
            if self.uv == True:
                e_uv_interact = self.v_to_e.weight[history]
                e_uv = self.u_to_e.weight[nodes[i]]
            else:
                e_uv_interact = self.u_to_e.weight[history]
                e_uv = self.v_to_e.weight[nodes[i]]
            att_w = self.att(e_uv_interact, e_uv, num_interact)
            att_history = torch.mm(e_uv_interact.t(), att_w)
            att_history = att_history.t()
            embed_matrix[i] = att_history
        neigh_feature = embed_matrix
        return neigh_feature

class SimilarLearner_Aggregator(nn.Module):

    def __init__(self, u_to_e, social_lists, embed_dim, cuda="cpu"):
        super(SimilarLearner_Aggregator, self).__init__()
        self.u_to_e = u_to_e
        self.social_lists = social_lists
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)
        self.device = cuda

    def forward(self, nodes):
        neighs = []
        for node in nodes:
            neighs.append(self.social_lists[int(node)])
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            i_social = neighs[i]
            num_neighs = len(i_social)
            e_u_social = self.u_to_e.weight[list(i_social)]
            e_u = self.u_to_e.weight[nodes[i]]
            # e_u.shape:64, e_u_social.shape[1, 64]
            att_w = self.att(e_u_social, e_u, num_neighs)
            att_history = torch.mm(e_u_social.t(), att_w).t()
            # print('e_u.shape:{}, e_u_social.shape:{}'.format(e_u.shape, e_u_social.shape))
            embed_matrix[i] = att_history
        social_feature = embed_matrix
        return social_feature

class LearnerC_Aggregator(nn.Module):
    def __init__(self, u_to_e, g_to_e, a_to_e, user_gender_list, user_age_list, embed_dim):
        super(LearnerC_Aggregator, self).__init__()
        self.u_to_e = u_to_e
        self.g_to_e = g_to_e
        self.a_to_e = a_to_e
        self.user_gender_list = user_gender_list
        self.user_age_list = user_age_list
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)
        self.device = 'cpu'
        self.linear = nn.Linear(self.embed_dim * 3, self.embed_dim)

    def forward(self, nodes):
        node_gender = []
        node_age = []
       
        for node in nodes:
            node_gender.append(self.user_gender_list[int(node)])
            node_age.append(self.user_age_list[int(node)])

        # e_g/e_a/e_u shape:[128, 64]
        e_g = self.g_to_e.weight[node_gender]
        e_a = self.a_to_e.weight[node_age]
        e_u = self.u_to_e.weight[nodes]
        # print('e_g.shape:{}, e_a.shape:{}'.format(e_g.shape, e_a.shape))

        # 共128个nodes，每个node上有64个特征。
        embed_matrix_eg = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        embed_matrix_ea = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        embed_matrix_eu = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        
        for i in range(len(nodes)):
            e_g_1 = e_g[i].view(1, 32)
            e_g_2 = e_g[i]
            att_w = self.att(e_g_1, e_g_2, 1)
            att_history = torch.mm(e_g_1.t(), att_w).t()
            embed_matrix_eg[i] = att_history
        for i in range(len(nodes)):
            e_u_1 = e_u[i].view(1, 32)
            e_u_2 = e_u[i]
            att_w = self.att(e_u_1, e_u_2, 1)
            att_history = torch.mm(e_u_1.t(), att_w).t()
            embed_matrix_eu[i] = att_history
        for i in range(len(nodes)):
            e_a_1 = e_a[i].view(1, 32)
            e_a_2 = e_a[i]
            att_w = self.att(e_a_1, e_a_2, 1)
            att_history = torch.mm(e_a_1.t(), att_w).t()
            embed_matrix_ea[i] = att_history
        x = torch.cat((embed_matrix_eg, embed_matrix_ea, embed_matrix_eu), 1)
        self_feats = F.relu(self.linear(x))
        return self_feats
 
class Learner_Encoder(nn.Module):

    def __init__(self, args, uv_aggregator, social_aggregator, user_self, embed_dim, cuda="cpu"):
        super(Learner_Encoder, self).__init__()
        self._parse_args(args)
        self.uv_aggregator = uv_aggregator
        self.social_aggregator = social_aggregator
        self.user_self = user_self
        self.embed_dim = embed_dim
        self.linear = nn.Linear(3 * self.embed_dim, self.embed_dim)
        self.device = cuda

    def _parse_args(self, args):
        self.interact_weight = args.interact_weight
        self.neigh_weight = args.neigh_weight
        self.self_weight = args.self_weight

    def forward(self, nodes):
        self_feature = self.user_self.forward(nodes)
        interact_feature = self.uv_aggregator.forward(nodes)
        neigh_feature = self.social_aggregator.forward(nodes)
        self_feature = self.self_weight * self_feature
        interact_feature = self.interact_weight * interact_feature
        neigh_feature = self.neigh_weight * neigh_feature
        combined = torch.cat([self_feature, interact_feature, neigh_feature], dim=1)
        encoder_u = F.relu(self.linear(combined))
        return encoder_u

