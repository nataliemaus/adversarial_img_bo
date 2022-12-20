import torch 

class AE(torch.nn.Module):
    def __init__(
        self,
        input_shape=768,
        n_layers=5,
    ):
        super().__init__()
        enc_layers = []
        dec_layers = []
        for i in range(n_layers):
            if i > 0:
                enc_layers.append(torch.nn.ReLU())
                dec_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.Linear(input_shape//(2**i), input_shape//(2**(i+1))))
            dec_layers.append(torch.nn.Linear(input_shape//(2**(n_layers-i)), input_shape//(2**(n_layers-i-1))))
        self.encoder = torch.nn.Sequential(*enc_layers)
        self.decoder = torch.nn.Sequential(*dec_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x