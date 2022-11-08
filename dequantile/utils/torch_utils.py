import torch
from torch import nn


class TupleDataset(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.tensors = args

    def __len__(self):
        return min([len(tensor) for tensor in self.tensors])

    def __getitem__(self, item):
        return [tensor[item] for tensor in self.tensors]


class GradsOff:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.requires_grad_(False)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.model.requires_grad_(True)


class CudaDefault:

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()

    def __enter__(self):
        if self.cuda_available:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.set_default_tensor_type(torch.FloatTensor)


def mask_data(mask, *args):
    return [data[mask] for data in args]


def shuffle_tensors(*args):
    mx = torch.randperm(len(args[0]), device=torch.device('cpu'))
    return mask_data(mx, *args)


def no_more_grads(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


class WeightedClassificationLoss:

    def __init__(self, loss):
        self.loss = loss(reduction='none')

    def __call__(self, predictions, labels, weight=None):
        loss = self.loss(predictions, labels)
        if weight is not None:
            loss = loss.view(-1) * weight.view(-1)
        return loss


class MSELoss(WeightedClassificationLoss):

    def __init__(self):
        super(MSELoss, self).__init__(nn.MSELoss)
