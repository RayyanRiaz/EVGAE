import os
import pickle
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter


class Analyser:
    def __init__(self, **kwargs):
        self.options = kwargs
        LOGS_DIR_TIMESTAMP = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.options["logdir"] = os.path.join((self.options["logdir"] if "logdir" in self.options else "results"), LOGS_DIR_TIMESTAMP + (self.options["comment"] if "comment" in self.options else ""))
        if not os.path.exists(self.options["logdir"]):
            os.makedirs(self.options["logdir"])
        self.writer = SummaryWriter(log_dir=self.options["logdir"], flush_secs=self.options["flush_secs"] if "flush_secs" in self.options else 120)
        self.scalars = {}

    def refresh(self):
        self.scalars = {}

    def add_to_scalars(self, **kwargs):
        for k in kwargs.keys():
            self.scalars[k] = (self.scalars[k] + kwargs[k]) if k in self.scalars else kwargs[k]

    def write_scalars(self, epoch):
        for k in self.scalars.keys():
            self.writer.add_scalar(k, self.scalars[k], epoch)

    def save_weights(self, model, epoch):
        torch.save(model.state_dict(), self.options["logdir"] + "/model_weights_" + str(epoch) + ".pk")

    def update_str(self, epoch, epoch_time=None):
        losses_str = (": {:4.5f}\t\t".join(self.scalars.keys()) + ": {:4.5f}").format(*[x for x in self.scalars.values()])
        epoch_str = "Epoch:{:3d}\t[time={:3.2f}s]\t\t".format(epoch, epoch_time) if epoch_time is not None else "Epoch:{:3d}\t\t".format(epoch)
        return epoch_str + losses_str

    def write_scalars_to_file(self, epoch, filename=None):
        filename = "results" if filename is None else filename
        with open(self.writer.get_logdir() + "/" + filename + ".txt", 'a') as f:
            print(self.update_str(epoch), file=f)

        try:
            with open(self.writer.get_logdir() + "/" + filename + ".pk", 'rb') as f:
                current_dict: dict = pickle.load(f)
            current_dict.update({epoch: self.scalars})
        except:
            current_dict = {epoch: self.scalars}
        finally:
            with open(self.writer.get_logdir() + "/" + filename + ".pk", 'wb') as f:
                pickle.dump(current_dict, f)


class VAEAnalyser(Analyser):
    def __init__(self, **kwargs):
        super(VAEAnalyser, self).__init__(**kwargs)

    def write_scalars_per_component(self, epoch, values, tag):
        scalars_dict = {str(k): v for k, v in dict(enumerate(values.detach().cpu().numpy())).items()}
        self.writer.add_scalars(tag, scalars_dict, epoch)

    def write_unit_activity_scalars(self, epoch):
        self.model.cpu()
        x, y = self.DL.dataset.tensors
        z, mu, logvar = self.model.encode(x)

        x_hat = self.model.decoder(z)
        l_x = self.model.l_x(x, x_hat)
        deviations_from_x_hat = []
        deviations_from_x = []
        for i in range(z.shape[1]):
            mask = torch.ones(z.shape[1])
            mask[i] = 0
            z_masked = z * mask[None, :]
            x_hat_masked = self.model.decoder(z_masked)
            deviation_from_x_hat = F.binary_cross_entropy(x_hat_masked, x_hat, reduction='none')
            deviation_from_x = F.binary_cross_entropy(x_hat_masked, x, reduction='none')
            # TODO: also compare the deviation with original x instead of x_hat
            # deviations_from_x_hat.append(deviation_from_x_hat.sum(1).mean().item())
            # deviations_from_x.append(deviation_from_x.sum(1).mean().item())
            deviations_from_x_hat.append(torch.abs(deviation_from_x_hat - l_x).sum(1).mean().item())
            deviations_from_x.append(torch.abs(deviation_from_x - l_x).sum(1).mean().item())

        deviations_from_x_hat_dict = {str(k): v for k, v in zip(list(range(mu.shape[1])), deviations_from_x_hat)}
        deviations_from_x_dict = {str(k): v for k, v in zip(list(range(mu.shape[1])), deviations_from_x)}
        self.summary_writer.add_scalars("deviations from x-hat", deviations_from_x_hat_dict, epoch)
        self.summary_writer.add_scalars("deviations from x", deviations_from_x_dict, epoch)


if __name__ == "__main__":
    analyser = Analyser(**{"a": "aaaa", "b": 123})
    print("initialized")
