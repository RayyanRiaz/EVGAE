import pickle
import time

import torch
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

    l_a_pos, l_a_neg = model.recon_loss_without_reduction(z, train_pos_edge_index, neg_edge_index)
    l_a = l_a_pos.mean() + l_a_neg.mean()

    l_kl, l_q = model.kl_losses(x, train_pos_edge_index)
    # l_kl *= (1 / data.num_nodes)
    # l_x *= 0
    l_x = 0
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
    channels_list = [16]
    allow_shifts = True

    Y = data.y
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_mask, test_mask = data.train_mask, data.test_mask
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = EpitomeVGAE.split_edges(data)
    x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)

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
                        cls_acc, nmi, ari, f1 = 0, 0, 0, 0
                        pre = Y.numpy()
                        auc, ap, class_acc = test(data.test_pos_edge_index, data.test_neg_edge_index)
                        KLD = -0.5 * torch.mean(1 + model.__logvar__ - model.__mu__.pow(2) - model.__logvar__.exp(), dim=0)

                        epoch_vars_dict = {"l_a": (l_a), "l_x": l_x, "kl_loss": l_kl.sum(1).mean(), 'l_q': l_q.mean(), "auc": auc,
                                           "ap": ap,
                                           "cls_acc": cls_acc, 'class_acc': class_acc, "nmi": nmi, "f1": f1, "ari": ari}
                        active_units = torch.where(((model.__mu__ - model.__mu__.mean(0)[None, :]) ** 2).mean(0) > 1e-2)[0]
                        analyser.add_to_scalars(**{
                            'avg_kll(au)': l_kl.mean(0)[active_units].mean() if not torch.isnan(l_kl.mean(0)[active_units].mean()) else torch.tensor(
                                0).float(),
                            'avg_kld(au)': KLD[active_units].mean() if not torch.isnan(KLD[active_units].mean()) else torch.tensor(0).float(),
                            'num_Au': active_units.shape[0],
                        })

                        analyser.add_to_scalars(**epoch_vars_dict)
                        print(analyser.update_str(epoch, epoch_time=(time.time() - start_time)))

                        analyser.write_scalars_per_component(epoch, KLD, "KLD_i")
                        analyser.write_scalars_per_component(epoch, l_kl.mean(0), "L_KLD_i")
                        analyser.write_scalars_per_component(epoch, model.__mu__.mean(0), "Mu_i")
                        analyser.write_scalars_per_component(epoch, torch.exp(model.__logvar__ / 2).mean(0), "Var_i")
                        analyser.write_scalars_per_component(epoch, ((model.__mu__ - model.__mu__.mean(0)[None, :]) ** 2).mean(0), "Unit Activity")

                        plots_data[epoch] = {
                            "kll": l_kl.mean(0).detach().cpu().numpy(),
                            "kld": KLD.detach().cpu().numpy(),
                            "mu": model.__mu__.mean(0).detach().cpu().numpy(),
                            "logvar": model.__logvar__.mean(0).detach().cpu().numpy(),
                            "au": ((model.__mu__ - model.__mu__.mean(0)[None, :]) ** 2).mean(0).detach().cpu().numpy()
                        }

                        analyser.writer.add_embedding(
                            mat=model.__mu__.detach().cpu(), tag="mu", global_step=epoch, metadata_header=["true", "pred"],
                            metadata=torch.cat((Y[:, None], torch.from_numpy(pre)[:, None]), dim=1).detach().cpu().tolist()
                        )

                        model.train()
                with open("plots_data.pickle", "wb") as f:
                    pickle.dump(plots_data, f)
