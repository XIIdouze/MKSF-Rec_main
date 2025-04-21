import torch
import torch.nn as nn
import numpy as np
from User import LearnerKnowledge_Aggregator
from User import LearnerC_Aggregator
from User import SimilarLearner_Aggregator
from User import Learner_Encoder
from Knowledge import  Knowledge_Encoder
from Item import RippleNet
from data_loader import load_data
import torch.utils.data
import argparse
import os
from model import MKSFRec,test,train

def main():

    np.random.seed(555)
    parser = argparse.ArgumentParser(description='MKSFRec_model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for training')
    parser.add_argument('--embed_dim', type=int, default=32, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1024, metavar='N', help='test batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--n_memory', type=int, default=16, help='size of ripple set for each hop')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                        help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')
    parser.add_argument('--use_cuda', type=bool, default=False, help='whether to use gpu')  # default = True

    parser.add_argument('--neigh_weight', type=float, default=0.3, help='weight of the KGE term/ social term')
    parser.add_argument('--interact_weight', type=float, default=0.3, help='weight of the interaction term')
    parser.add_argument('--self_weight', type=float, default=0.4, help='weight of the self term')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = True
    if torch.cuda.is_available():
        use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim

    data_file = open('./data/30%datasets.txt', 'r')
    a = data_file.read()
    data = eval(a)
    history_u_lists, history_v_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_lists, user_gender_list, user_age_list = data
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = history_u_lists.__len__()
    num_items = max(history_v_lists)
    num_genders = 2
    num_ages = 7

    u_to_e = nn.Embedding(num_users+1, embed_dim).to(device)
    v_to_e = nn.Embedding(num_items+1, embed_dim).to(device)
    g_to_e = nn.Embedding(num_genders, embed_dim).to(device)
    a_to_e = nn.Embedding(num_ages, embed_dim).to(device)

    # get KG
    n_entity, n_relation, ripple_set, history_not_kg_v, item_index_old_to_new = load_data(args)

    # user feature
    u_interact_feature = LearnerKnowledge_Aggregator(v_to_e, u_to_e, history_u_lists, embed_dim, uv=True)
    u_social_feature = SimilarLearner_Aggregator(u_to_e, social_lists, embed_dim)
    u_self_feature = LearnerC_Aggregator(u_to_e, g_to_e, a_to_e, user_gender_list, user_age_list, embed_dim)
    # combine
    encoder_u = Learner_Encoder(args, u_interact_feature, u_social_feature, u_self_feature, embed_dim)

    # item feature:
    v_interact_feature = LearnerKnowledge_Aggregator(v_to_e, u_to_e, history_v_lists, embed_dim, uv=False)
    # KG feature
    v_kg_feature = RippleNet(args, v_to_e, n_entity, n_relation, ripple_set, history_not_kg_v, item_index_old_to_new)
    # combine
    encoder_v =Knowledge_Encoder(args, v_to_e, v_interact_feature, v_kg_feature,  embed_dim)

    # model
    social_ripple_net = MKSFRec(encoder_u, encoder_v).to(device)
    # Add L2 regularization to the optimizer
    l2_lambda = 0.0000001  # Set your desired L2 regularization strength
    optimizer = torch.optim.Adam(social_ripple_net.parameters(), lr=args.lr, weight_decay=l2_lambda)
    result_list = []
    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(social_ripple_net, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = test(social_ripple_net, device, test_loader)
        result_list.append((epoch, expected_rmse, mae))
        # early stopping
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 5:
            print('early stopping')
            break
        with open("30%_h2_b32_r0.01_l2=0.0000001.txt", "w") as file:
            for epoch, rmse, mae in result_list:
                file.write(f"Epoch {epoch}: RMSE = {rmse:.4f}, MAE = {mae:.4f}\n")

if __name__ == "__main__":
    main()
