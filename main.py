import pickle
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import *
from sklearn.mixture import GaussianMixture
from torch.distributions import kl_divergence, Normal
from torch_geometric.utils import negative_sampling

from analysis.base import VAEAnalyser
from helpers import map_labels, get_planetoid_dataset
from model import EpitomeVGAE

torch.manual_seed(12345)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    # x_hat = model.node_decoder(z, train_pos_edge_index)
    x_hat = model.node_decoder(z)
    neg_edge_index = negative_sampling(train_pos_edge_index, x.size(0))
    # l_a_pos, l_a_neg = model.recon_loss_without_reduction(z, train_pos_edge_index, neg_edge_index)
    # l_a_pos, l_a_neg = model.recon_loss_without_reduction(z, x, train_pos_edge_index, neg_edge_index)
    l_a_pos, l_a_neg = model.recon_loss_without_reduction(z, train_pos_edge_index, neg_edge_index)
    l_a = l_a_pos.mean() + l_a_neg.mean()
    # l_a = model.recon_loss(z, train_pos_edge_index)

    # l_x_p = F.binary_cross_entropy(x_hat[train_pos_edge_index[0]], x[train_pos_edge_index[1]], reduction='none')
    # l_x_n = F.binary_cross_entropy(x_hat[neg_edge_index[0]], 1 - x[neg_edge_index[1]], reduction='none')
    # l_x = l_x_p.sum(1).mean() + l_x_n.sum(0).mean()
    ## this one!
    # l_x = F.binary_cross_entropy(x_hat, x, reduction='none').sum(1).mean()
    l_kl, l_q = model.kl_losses(x, train_pos_edge_index)
    # l_kl *= (1 / data.num_nodes)
    # l_kl *= (1 / data.num_nodes)
    # l_kl /= 100000
    l_kl *= 0
    # l_x *= 0
    l_x = 0
    # l_q *= 0
    loss = l_a + l_kl.sum(1).mean() + l_q.mean() + l_x
    loss.backward()
    optimizer.step()

    return z, neg_edge_index, l_a, l_a_pos, l_a_neg, l_x, l_kl, l_q


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)

    class_acc = model.test_model(z[train_mask], Y[train_mask], z[test_mask], Y[test_mask], max_iter=150)
    auc, ap = model.test(z, pos_edge_index, neg_edge_index)

    return auc, ap, class_acc


if __name__ == "__main__":
    args = {
        "dataset": "Cora",  # ['Cora', 'CiteSeer', 'PubMed']
        "cls": 7

    }

    dataset = get_planetoid_dataset(args["dataset"])
    data = dataset[0]
    print(data.num_nodes)
    # channels_list = [int(sys.argv[1])] if sys.argv[1] else [2**i for i in range(3, 10)]
    # channels_list = [2**i for i in range(4, 10)]
    channels_list = [16]
    # allow_shifts = sys.argv[2] and sys.argv[2] == 'y'
    allow_shifts = True


    Y = data.y
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_mask, test_mask = data.train_mask, data.test_mask
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = EpitomeVGAE.split_edges(data)
    x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
    # neg_edge_index = negative_sampling(train_pos_edge_index, x.size(0))

    for channels in channels_list:
        # num_epitomes are power of 2, till we reach num_channels
        # num_epitomes_list = [2 ** i for i in range(0, int(math.log2(channels)))]
        num_epitomes_list = [1]

        for num_epitomes in num_epitomes_list:
            # shifts are multiple of 2, till we reach mid of ones per epitome.
            # eg for 16 channels and 2 epitomes, we have 8 ones per epitome => shifts=[0, 2, 4, 6].
            # We have added as exception when num_epitomes=1 or when we have 2 ones per epitome
            if allow_shifts:
                if num_epitomes == 1:
                    shifts_list = [0]
                elif num_epitomes == channels / 2:
                    shifts_list = [1]
                else:
                    # shifts_list = [i * 2 for i in range(int(channels / 2 / num_epitomes))]
                    # shifts_list = [i * 2 for i in range(min(10, int(channels / 2 / num_epitomes)))]
                    shifts_list = [1]
            else:
                shifts_list = [0]
            for shifts in shifts_list:
                print("channels: {}, epitomes: {}, shifts: {}".format(channels, num_epitomes, shifts))
                model = EpitomeVGAE(x_dim=data.num_features, z_dim=channels, num_epitomes=num_epitomes, shifts=shifts).to(dev)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                #######################################################
                # analyser = VAEAnalyser(**{"comment": "z_{}__e_{}__s_{}".format(channels, num_epitomes, shifts)})
                analyser = VAEAnalyser(**{"logdir": 'e1res', "comment": "z_{}".format(channels)})
                plots_data = {}
                for epoch in range(0, 5000):
                    start_time = time.time()
                    analyser.refresh()
                    z, neg_edge_index, l_a, l_a_pos, l_a_neg, l_x, l_kl, l_q = train()
                    if epoch % 10 == 0:
                        model.eval()
                        # try:
                        #     gmm = GaussianMixture(n_components=args["cls"], covariance_type='diag', n_init=5)
                        #     pre = gmm.fit_predict(model.__mu__.detach().cpu().numpy())
                        #     y_map = map_labels(pre, Y.cpu().numpy())
                        #     pre = y_map[1][pre]
                        #     # cls_acc = cluster_acc(pre, Y.cpu().numpy())[0] * 100
                        #     cls_acc = accuracy_score(Y.cpu().numpy(), pre) * 100
                        #     nmi = normalized_mutual_info_score(Y.cpu().numpy(), pre, average_method="arithmetic") * 100
                        #     f1 = f1_score(Y.cpu().numpy(), pre, average="macro") * 100
                        #     ari = adjusted_rand_score(Y.cpu().numpy(), pre) * 100
                        # except:
                        #     cls_acc, nmi, ari, f1 = 0, 0, 0, 0
                        cls_acc, nmi, ari, f1 = 0, 0, 0, 0
                        pre = Y.numpy()
                        auc, ap, class_acc = test(data.test_pos_edge_index, data.test_neg_edge_index)
                        KLD = -0.5 * torch.mean(1 + model.__logvar__ - model.__mu__.pow(2) - model.__logvar__.exp(), dim=0)

                        epoch_vars_dict = {"l_a": (l_a), "l_x": l_x, "kl_loss": l_kl.sum(1).mean(), 'l_q': l_q.mean(), "auc": auc,
                                           "ap": ap,
                                           "cls_acc": cls_acc, 'class_acc': class_acc, "nmi": nmi, "f1": f1, "ari": ari}
                        active_units = torch.where(((model.__mu__ - model.__mu__.mean(0)[None, :])**2).mean(0) > 1e-2)[0]
                        analyser.add_to_scalars(**{
                            'avg_kll(au)': l_kl.mean(0)[active_units].mean() if not torch.isnan(l_kl.mean(0)[active_units].mean()) else torch.tensor(0).float(),
                            'avg_kld(au)': KLD[active_units].mean() if not torch.isnan(KLD[active_units].mean()) else torch.tensor(0).float(),
                            'num_Au': active_units.shape[0],
                        })

                        analyser.add_to_scalars(**epoch_vars_dict)
                        # analyser.write_scalars(epoch)
                        # analyser.save_weights(model, epoch)
                        # analyser.write_scalars_to_file(epoch, filename=("z_{}__e_{}__s_{}".format(channels, num_epitomes, shifts)))
                        print(analyser.update_str(epoch, epoch_time=(time.time() - start_time)))

                        analyser.write_scalars_per_component(epoch, KLD, "KLD_i")
                        analyser.write_scalars_per_component(epoch, l_kl.mean(0), "L_KLD_i")
                        analyser.write_scalars_per_component(epoch, model.__mu__.mean(0), "Mu_i")
                        analyser.write_scalars_per_component(epoch, torch.exp(model.__logvar__ / 2).mean(0), "Var_i")
                        analyser.write_scalars_per_component(epoch, ((model.__mu__ - model.__mu__.mean(0)[None, :])**2).mean(0), "Unit Activity")

                        plots_data[epoch] = {
                            "kll": l_kl.mean(0).detach().cpu().numpy(),
                            "kld": KLD.detach().cpu().numpy(),
                            "mu": model.__mu__.mean(0).detach().cpu().numpy(),
                            "logvar": model.__logvar__.mean(0).detach().cpu().numpy(),
                            "au": ((model.__mu__ - model.__mu__.mean(0)[None, :])**2).mean(0).detach().cpu().numpy()
                        }
                        #
                        #
                        # # analyser.writer.add_custom_scalars_multilinechart(["Var_i"])

                        analyser.writer.add_embedding(
                            mat=model.__mu__.detach().cpu(), tag="mu", global_step=epoch, metadata_header=["true", "pred"],
                            metadata=torch.cat((Y[:, None], torch.from_numpy(pre)[:, None]), dim=1).detach().cpu().tolist()
                        )



                        #############################
                        # a = torch.cat((torch.ones(train_pos_edge_index.shape[1], 1), torch.zeros(neg_edge_index.shape[1], 1)), dim=0).squeeze().cuda()
                        # a_hat_pos = model.decoder(z, train_pos_edge_index)
                        # a_hat_neg = model.decoder(z, neg_edge_index)
                        # # a_hat = torch.cat((a_hat_pos[:, None], a_hat_neg[:, None]), dim=0).squeeze()
                        # # l_a_without_reduction = torch.cat((l_a_pos[:, None], l_a_neg[:, None]), dim=0).squeeze()
                        # # deviations_from_x_hat = []
                        # deviations_from_x = []
                        # for i in range(z.shape[1]):
                        #     mask = torch.ones(z.shape[1]).cuda()
                        #     mask[i] = 0
                        #     # mask[(i + 1) % z.shape[1]] = 0
                        #     # mask[(i + 2) % z.shape[1]] = 0
                        #
                        #     z_masked = z * mask[None, :]
                        #     a_hat_masked_pos = model.decoder(z_masked, train_pos_edge_index)
                        #     a_hat_masked_neg = model.decoder(z_masked, neg_edge_index)
                        #     # a_hat_masked = torch.cat((a_hat_masked_pos[:, None], a_hat_masked_neg[:, None]), dim=0).squeeze()
                        #
                        #     l_a_masked_pos = -torch.log(a_hat_masked_pos + 1e-15)
                        #     l_a_masked_neg = -torch.log(1-a_hat_masked_neg + 1e-15)
                        #     deviation_from_x = torch.abs(l_a_masked_pos - l_a_pos).mean() + torch.abs(l_a_masked_neg - l_a_neg).mean()
                        #     # deviation_from_x = F.binary_cross_entropy(a_hat_masked, a_hat.detach(), reduction='none')
                        #     # deviation_from_x = (-torch.log(a_hat_masked_pos + 1e-15).mean() - torch.log(1 - a_hat_masked_neg + 1e-15).mean())
                        #     deviations_from_x.append(deviation_from_x.item())
                        #     # deviations_from_x.append(deviation_from_x.mean().item())
                        #
                        # deviations_from_x_dict = {str(k): v for k, v in zip(list(range(z.shape[1])), deviations_from_x)}
                        # analyser.writer.add_scalars("deviations from x", deviations_from_x_dict, epoch)



                        # pairs_of_components = torch.combinations(torch.arange(0, z.shape[1]), 2)
                        # kl_bw_components = {}
                        # for p in pairs_of_components:
                        #     kl_bw_components[str(p)] = kl_divergence(
                        #         Normal(model.__mu__[:, p[0]], torch.exp(model.__logvar__[:, p[0]] / 2)),
                        #         Normal(model.__mu__[:, p[1]], torch.exp(model.__logvar__[:, p[1]] / 2))
                        #     ).mean().item()
                        # analyser.writer.add_scalars("kl_bw_components", kl_bw_components, epoch)
                        #

                        model.train()
                with open("plots_data.pickle", "wb") as f:
                    pickle.dump(plots_data, f)
