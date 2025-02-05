from ccbdl.network.nlrl import NLRL_AO as NLRL
import torch

device = "cuda"

criterion = torch.nn.BCELoss()
net = NLRL(5, 1).cuda()
ins = torch.ones(16, 5).to(device)

targets = torch.ones(16, 1).to(device)
print(criterion(net(ins), targets))



