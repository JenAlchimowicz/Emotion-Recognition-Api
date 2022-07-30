import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pytest

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir)))
from architectures.DAN import DAN
from architectures.ED_model import ED_model

from test_utils import _train_step


@pytest.mark.parametrize('input_shape', [
    ([1,3,224,224]),
    ([5,3,224,224]),
    pytest.param([3,224,224], marks=pytest.mark.xfail),
    pytest.param([5,5], marks=pytest.mark.xfail),
    pytest.param([1,1,224,224], marks=pytest.mark.xfail),
    pytest.param([3,3,512,5]),
])
def test_DAN_output_shape(input_shape):
    model = DAN(num_head=4, num_class=8, pretrained=False)
    model.eval()

    x = torch.rand(input_shape, dtype=torch.float)
    output = model(x)

    assert output.shape == torch.Size([input_shape[0],8])



@pytest.mark.parametrize('input_shape', [
    ([1,1,48,48]),
    ([5,1,48,48]),
    pytest.param([3,48,48], marks=pytest.mark.xfail),
    pytest.param([5,5], marks=pytest.mark.xfail),
    pytest.param([1,3,48,48], marks=pytest.mark.xfail),
    pytest.param([1,1,50,20], marks=pytest.mark.xfail),
])
def test_ED_output_shape(input_shape):
    model = ED_model(in_channels=1, out_channels=7)
    model.eval()

    x = torch.rand(input_shape, dtype=torch.float)
    output = model(x)

    assert output.shape == torch.Size([input_shape[0],7])



@pytest.mark.parametrize('input_shape', [
    ([2,3,224,224]),
    ([5,3,224,224]),
])
def test_DAN_variable_change(input_shape):
    '''
    Check if all parameters in the DAN model DO change

    Will rise an error if some parameters DO NOT change, thus were ommited in training. This can happen if:
        - We define a layer in a model but never use it
        - Incorrect initialisation
        - Other issues
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DAN(num_head=4, num_class=8, pretrained=False)
    
    inputs = Variable(torch.rand(input_shape, dtype=torch.float))
    targets = Variable(torch.randint(0,8,(input_shape[0],)))

    # get initial parameters
    params = [ np for np in model.named_parameters() if np[1].requires_grad ]
    initial_params = [ (name, p.clone()) for (name, p) in params ]

    # run a training step
    loss_fn = F.cross_entropy
    optim = torch.optim.Adam(model.parameters())
    batch = [inputs, targets]
    _train_step(model, loss_fn, optim, batch, device)

    # Check if variables changed
    for (_, p0), (name, p1) in zip(initial_params, params):
        try:
            assert not torch.equal(p0.to(device), p1.to(device))
        except:
            raise ValueError(f'Variable {name} did not change!')



@pytest.mark.parametrize('loss_fn', [
    (F.cross_entropy)
])
def test_DAN_loss(loss_fn):
    '''
    Check that a loss is never 0.

    Will rise an error is loss is calculated to be exactly 0: This can happen:
        - In some unexpected edge cases (e.g. applying softmax cross entropy to an output with dimension 1)
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DAN(num_head=4, num_class=8, pretrained=False)
    
    input_shape = [2,3,224,224]
    inputs = Variable(torch.rand(input_shape, dtype=torch.float))
    targets = Variable(torch.randint(0,8,(input_shape[0],)))

    # get model outputs
    inputs = inputs.to(device)
    model.to(device)
    model.eval()
    output = model(inputs)

    loss = loss_fn(output, targets)

    try:
        assert loss != 0
    except:
        raise AssertionError('Loss is equal to 0!')


@pytest.mark.parametrize('input_shape', [
    ([2,3,224,224])
])
def test_DAN_never_nan(input_shape):
    '''
    Check that the model does not output NaN values

    Will rise an error if at least one ov the values in the output tensor are NaN. This can happen:
        - In case of exploding gradients
        - In case of other unexpected edge cases
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DAN(num_head=4, num_class=8, pretrained=False)
    
    inputs = Variable(torch.rand(input_shape, dtype=torch.float))
    targets = Variable(torch.randint(0,8,(input_shape[0],)))

    # get model outputs
    inputs = inputs.to(device)
    model.to(device)
    model.eval()
    output = model(inputs)

    try:
        assert not torch.isnan(output).byte().any()
    except:
        raise AssertionError('There was a NaN value in the output tensor')