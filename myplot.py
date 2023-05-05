import matplotlib.pyplot as plt
import torch


def woplot(data, t,type='--'):
    plt.plot(t, data[:, 0], label='S',linestyle=type)
    plt.plot(t, data[:, 1], label='I',linestyle=type)
    plt.plot(t, data[:, 2], label='R',linestyle=type)
    plt.plot(t, data[:, 3], label='D',linestyle=type)
    plt.plot(t, data[:, 4], label='N',linestyle=type)
    plt.legend()
    plt.show()


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
