import torch

from architectures.DAN import DAN

model = DAN(num_head=4, num_class=8, pretrained=False)

# for np in model.named_parameters():
#     print(np)

for i in range(20):
    print(torch.randint(0,3, (5,)))