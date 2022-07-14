import torch
import pytest

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir)))
from architectures.DAN import DAN
from architectures.my_model import ED_model


@pytest.mark.parametrize('input_shape', [
    ([1,3,224,224]),
    ([5,3,224,224]),
    pytest.param([3,224,224], marks=pytest.mark.xfail),
    pytest.param([5,5], marks=pytest.mark.xfail),
    pytest.param([1,1,224,224], marks=pytest.mark.xfail),
    pytest.param([3,3,512,5]),
])
def test_DAN(input_shape):
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
def test_ED(input_shape):
    model = ED_model(in_channels=1, out_channels=7)
    model.eval()

    x = torch.rand(input_shape, dtype=torch.float)
    output = model(x)

    assert output.shape == torch.Size([input_shape[0],7])