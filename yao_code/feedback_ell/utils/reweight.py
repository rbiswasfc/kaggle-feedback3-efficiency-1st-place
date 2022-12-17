"""
@created by: heyao
@created at: 2022-10-27 01:32:59
"""
import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def preparer_weights_for_multi_label(labels, reweight, max_target=51, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    weights = []
    for i in range(len(labels[0])):
        label = [k[i] for k in labels]
        weights.append(np.array(_prepare_weights(label, reweight, max_target, lds, lds_kernel, lds_ks, lds_sigma)).reshape(-1, 1))
    weights = np.concatenate(weights, axis=1).tolist()
    return weights


def _prepare_weights(labels, reweight, max_target=51, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    value_dict = {x: 0 for x in range(max_target)}
    # labels = self.data[:, -1].tolist()
    # mbr
    for label in labels:
        value_dict[min(max_target - 1, int(label))] += 1
    if reweight == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
    if not len(num_per_label) or reweight == 'none':
        return None
    print(f"Using re-weighting: [{reweight.upper()}]")

    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights


if __name__ == '__main__':
    import pandas as pd

    from feedback_ell.utils import label_columns

    f = "/home/heyao/kaggle/feedback-english-lan-learning/input/feedback-prize-english-language-learning/train.csv"
    df = pd.read_csv(f)
    labels = df[label_columns[0]]
    weights = _prepare_weights(labels, reweight="sqrt_inv", max_target=9, lds=False)
    print(list(set(zip(labels, weights))))
