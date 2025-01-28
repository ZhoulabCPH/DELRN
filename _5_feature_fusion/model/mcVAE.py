import torch
from torch.distributions import Bernoulli, Categorical, MultivariateNormal, kl_divergence
import matplotlib.pyplot as plt
from torch.distributions import Normal, kl_divergence, constraints
from torch.distributions.utils import _standard_normal, broadcast_all
import numpy as np

def multiply_gaussians(mean1, var1, mean2, var2):
    """
	The Product of two Gaussian PDFs is a Scaled (un-normalized) Gaussian PDF.
	Here we provide only the proper (normalized) PDF.
	"""
    mean = (mean1 * var2 + mean2 * var1) / (var1 + var2)
    var = 1 / (1 / var1 + 1 / var2)

    return mean, var


def divide_gaussians(mean1, var1, mean2, var2):
    """
	gauss1 / gauss2
	"""
    var = 1 / (1 / var1 - 1 / var2)
    mean = mean1 + var * (1 / var2) * (mean1 - mean2)

    return mean, var


def cholesky_to_variance(diagonal, lower_diagonal):
    """
	Return Sigma = CC', where C is che Cholesky factor
	"""
    L = len(diagonal)
    C = torch.diag(diagonal)
    C[torch.ones(L, L).tril(-1) == 1] = lower_diagonal
    return torch.matmul(C, C.t())


def plot_covariance_ellipsoid(sigma):
    t = np.linspace(0, 2 * np.pi)
    if not isinstance(sigma, list):
        sigma = [sigma]
    for sigma_ in sigma:
        d, v = np.linalg.eig(sigma_)
        print(f'Sqrt(eig_values) = {d ** 0.5}')
        x = np.array([[np.cos(_), np.sin(_)] for _ in t])
        a = (v * d ** 0.5).dot(x.T).T
        plt.plot(a[:, 1], a[:, 0])
    plt.xlabel('ax 1')
    plt.ylabel('ax 0')
    plt.axis('equal')


def trilinear_covariance(n, var_diag, var_offdiag, device='cpu'):
    cm = torch.zeros(n, n).to(device)
    cm[torch.ones(n, n).diag().diag() == 1] = var_diag

    if var_offdiag:
        cm[torch.ones(n, n).tril(-1) == 1] = var_offdiag
        cm[torch.ones(n, n).triu(1) == 1] = var_offdiag

    return cm


def multivariate_prior(n, device='cpu', *args, **kwargs):
    # Same arguments as trilinear_covariance function
    cm = trilinear_covariance(n=n, device=device, *args, **kwargs)
    return MultivariateNormal(loc=torch.zeros(n).to(device), covariance_matrix=cm)


def p_to_prediction(p):
    if isinstance(p, list):
        return [p_to_prediction(_) for _ in p]

    if isinstance(p, Normal):
        pred = p.loc
    elif isinstance(p, Categorical):
        pred = p.logits.argmax(dim=1)
    elif isinstance(p, Bernoulli):
        pred = p.probs
    else:
        raise NotImplementedError

    return pred

def KL_log_uniform(mu, logvar):
	"""
	Paragraph 4.2 from:
	Variational Dropout Sparsifies Deep Neural Networks
	Molchanov, Dmitry; Ashukha, Arsenii; Vetrov, Dmitry
	https://arxiv.org/abs/1701.05369
	https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb
	"""
	log_alpha = compute_log_alpha(mu, logvar)
	k1, k2, k3 = 0.63576, 1.8732, 1.48695
	neg_KL = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) - k1
	return -neg_KL

def compute_kl(p1, p2=None, sparse=False):
	"""
	:param p1: Normal distribution with p1.loc.shape = (n_obs, n_lat_dims)
	:param p2: same as p1
	:param sparse:
	:return: scalar value
	"""
	if sparse:
		kl = KL_log_uniform(mu=p1.loc, logvar=p1.scale.pow(2).log())
	else:
		kl = kl_divergence(p1, p2)

	return kl.sum(1, keepdims=True).mean(0)

def compute_ll(p, x):
	"""
	:param p: Normal: p.loc.shape = (n_obs, n_feats)
	:param x:
	:return: log-likelihood compatible with the distribution p
	"""
	if isinstance(p, Normal):
		ll = p.log_prob(x).sum(1, keepdims=True)
	elif isinstance(p, Categorical):
		ll = p.log_prob(x.view(-1))
	elif isinstance(p, MultivariateNormal):
		ll = p.log_prob(x).unsqueeze(1)  # MultiVariate already sums over dimensions
	else:
		raise NotImplementedError

	return ll.mean(0)



class Normal(Normal):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(
            self,
            loc,
            scale,
            scale_m=None,
            split_sizes=None,
            *args, **kwargs,
    ):
        super().__init__(loc, scale, *args, **kwargs)

        self.scale_m = scale_m
        if scale_m is not None:
            self.loc, self.scale_m = broadcast_all(loc, scale_m)

        if split_sizes is None:
            try:
                self.split_sizes = (self.loc.shape[0],)
            except:
                pass
        else:
            assert sum(split_sizes) == self.loc.shape[0]
            self.split_sizes = split_sizes

    @property
    def stddev(self):
        if self.scale_m is None:
            return self.scale
        else:
            return self.variance.pow(0.5)

    @property
    def variance(self):
        if self.scale_m is None:
            return self.scale.pow(2)
        else:
            return self.scale.pow(2) + self.scale_m.pow(2)

    def __mul__(self, other):
        """
		The Product of two Gaussian PDFs is a Scaled (un-normalized) Gaussian PDF.
		Here we provide only the proper (normalized) PDF.
		"""
        if other == 1:
            return self

        mean, var = multiply_gaussians(self.mean, self.variance, other.mean, other.variance)

        return Normal(loc=mean, scale=var.pow(0.5))

    def __truediv__(self, other):

        if other == 1:
            return self

        mean, var = divide_gaussians(self.mean, self.variance, other.mean, other.variance)

        return Normal(loc=mean, scale=var.pow(0.5))

    def __pow__(self, power, modulo=None):
        assert isinstance(power, int)
        assert power >= 0
        if power is 0:
            return 1
        if power is 1:
            return self
        else:
            p = self
            for i in range(1, power):
                p *= self
            return p

    def kl_divergence(self, other):
        return kl_divergence(Normal(loc=self.loc, scale=self.stddev), other)

    def kl_divergence_rev(self, other):
        return kl_divergence(other, self)

    def kl_divergence_symm(self, other):
        return 0.5 * (self.kl_divergence(other) + self.kl_divergence_rev(other))

    def kl_from_log_uniform(self):
        return KL_log_uniform(mu=self.loc, logvar=self.scale.pow(2).log())


    def plot(self):
        if len(self.loc.shape) > 1:
            self._plot_n()
        else:
            x = self.sample((1000,)).sort(0)[0]
            p = self.log_prob(x).exp()
            plt.plot(x.detach().numpy(), p.detach().numpy(), '.')

    def _plot_n(self):
        for l, s in zip(self.loc, self.stddev):
            Normal(loc=l, scale=s).plot()


def compute_log_alpha(mu, logvar):
    # clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
    return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)


def compute_logvar(mu, log_alpha):
    return log_alpha + 2 * torch.log(torch.abs(mu) + 1e-8)


