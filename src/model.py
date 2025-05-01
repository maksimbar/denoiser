import torch.nn as nn


class RCED(nn.Module):
    def __init__(self, filters, kernels, in_ch=8, skip=True):
        super().__init__()
        assert len(filters) == len(kernels)
        assert len(filters) % 2 == 1, (
            "filters list must have odd length (incl. bottleneck)"
        )
        self.skip_enabled = skip
        self.n_layers = len(filters) // 2

        enc_f = filters[: self.n_layers]
        enc_k = kernels[: self.n_layers]
        bot_f = filters[self.n_layers]
        bot_k = kernels[self.n_layers]
        dec_f = filters[self.n_layers + 1 :]
        dec_k = kernels[self.n_layers + 1 :]

        self.encoder = nn.ModuleList()
        ch = in_ch
        for i in range(self.n_layers):
            f, k = enc_f[i], enc_k[i]
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(ch, f, k, padding=k // 2, bias=True),
                    nn.BatchNorm1d(f),
                    nn.ReLU(inplace=True),
                )
            )
            ch = f

        self.bottleneck = nn.Sequential(
            nn.Conv1d(ch, bot_f, bot_k, padding=bot_k // 2, bias=True),
            nn.BatchNorm1d(bot_f),
            nn.ReLU(inplace=True),
        )
        ch = bot_f

        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            f, k = dec_f[i], dec_k[i]
            self.decoder.append(
                nn.Sequential(
                    nn.Conv1d(ch, f, k, padding=k // 2, bias=True),
                    nn.BatchNorm1d(f),
                    nn.ReLU(inplace=True),
                )
            )
            ch = f

        final_in_ch = dec_f[-1] if self.n_layers > 0 else bot_f
        self.final = nn.Conv1d(final_in_ch, 1, 129, padding=64)

    def forward(self, x):
        skips = []
        h = x

        for i, blk in enumerate(self.encoder):
            h = blk(h)
            if self.skip_enabled and i % 2 == 0:
                skips.append(h)

        h = self.bottleneck(h)

        for i, blk in enumerate(self.decoder):
            h = blk(h)
            corresponding_encoder_idx = self.n_layers - 1 - i
            if self.skip_enabled and corresponding_encoder_idx % 2 == 0:
                s = skips.pop()
                h = h + s

        return self.final(h).squeeze(1)
