from autodp import rdp_acct, rdp_bank
import torch
import math


class PrivacyCostAnalysis():
    def __init__(self) -> None:
        pass

    def gswgan_cost(self, batch_size, n_parties, train_iters, noise_multiplier, delta=1e-3):
        """
        batch_size # lower is better
        delta : higher is better
        train_iters: lower is better
        """
        # subsampling rate 
        prob = 1./n_parties # lower is better
        # training iterations
        n_steps = train_iters # lower is better
        # noise scale
        sigma = noise_multiplier # higher is better
        func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

        acct = rdp_acct.anaRDPacct()
        acct.compose_subsampled_mechanism(func, prob, coeff=n_steps * batch_size)
        epsilon = acct.get_eps(delta)
        print("Privacy cost is: epsilon={}, delta={}".format(epsilon, delta))
        return epsilon
    
    def moments_acc(self, num_teachers, clean_votes, lap_scale, l_list):
        # clean_votes [bacthsize, 1]
        q = (2 + lap_scale * torch.abs(2*clean_votes - num_teachers)
            )/(4 * torch.exp(lap_scale * torch.abs(2*clean_votes - num_teachers)))

        update = []
        for l in l_list:
            a = 2*lap_scale*lap_scale*l*(l + 1)
            t_one = (1 - q) * torch.pow((1 - q) / (1 - math.exp(2*lap_scale) * q), l)
            t_two = q * torch.exp(2*lap_scale * l)
            t = t_one + t_two
            update.append(torch.clamp(t, max=a).sum())

        return torch.cuda.DoubleTensor(update)   


if __name__ == '__main__':
    a = PrivacyCostAnalysis().gswgan_cost(64, 20, 500, 10)
    