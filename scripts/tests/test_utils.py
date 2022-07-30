import torch

def _train_step(model, loss_fn, optim, batch, device):
    '''
    Run a training step on a model fiven a batch of data
    Arguments:
        - model [torch.nn.Module] - model to train
        - loss_fn [torch.nn.Module] - loss funcion from torch.nn.functional
        - optim [torch.optim.Optimizer] - instance of optimizer
        - batch [list] - a 2-element list [torch.Tensor, torch.Tensor] containing the inputs and labels, in this order
    Outputs:
        - model will have accumulated gradients
        - model weights should be changed according to the optimizer parameters
    '''

    model.train()
    model.to(device)

    # Forward pass
    optim.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    output = model(inputs)
    loss = loss_fn(output, targets)

    # Backward pass
    loss.backward()
    optim.step()