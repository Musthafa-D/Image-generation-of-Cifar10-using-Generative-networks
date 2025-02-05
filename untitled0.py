torch.nn.Conv2d(in_channels, hidden_channels, 4, 2, 1, bias=False), # 32
torch.nn.LeakyReLU(0.2, inplace=True),

torch.nn.Conv2d(hidden_channels, hidden_channels * 2, 4, 2, 1, bias=False), # 16
torch.nn.BatchNorm2d(hidden_channels * 2),
torch.nn.LeakyReLU(0.2, inplace=True),

torch.nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, 2, 1, bias=False), # 8
torch.nn.BatchNorm2d(hidden_channels * 4),
torch.nn.LeakyReLU(0.2, inplace=True),

torch.nn.Conv2d(hidden_channels * 4, hidden_channels * 8, 4, 2, 1, bias=False), # 4
torch.nn.BatchNorm2d(hidden_channels * 8),
torch.nn.LeakyReLU(0.2, inplace=True),

torch.nn.Conv2d(hidden_channels * 8, 12, 4, bias=False), # 1
torch.nn.Flatten(),
torch.nn.Sigmoid()
if final_layer.lower() == 'linear':
    torch.nn.Linear(8, 1)
    torch.nn.Sigmoid()
elif final_layer.lower() == 'nlrl':
    NLRL_AO(8, 1)
else:
    raise ValueError(
        f"Invalid value for final_layer: {final_layer}, it should be 'linear', or 'nlrl'")
