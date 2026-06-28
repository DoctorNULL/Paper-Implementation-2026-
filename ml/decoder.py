import torch
import torch.nn as nn


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.lstm1 = nn.LSTM(512, 512, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(512, 512, num_layers=1, batch_first=True)

        self.fc = nn.Linear(512, 512)
        self.out = nn.Linear(512, vocab_size)
        self.alpha_fc = nn.Linear(512, 1)

    def forward(self,
                v_center: torch.Tensor,
                v: torch.Tensor,
                prev_w: torch.Tensor = None,
                prev_v: torch.Tensor = None,
                prev_h1: torch.Tensor = None,
                prev_c1: torch.Tensor = None,
                prev_h2: torch.Tensor = None,
                prev_c2: torch.Tensor = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        # v_center (B, 1, 512), v (B, K + 1, 512), w (B, 1, 512), prev_v (B, 1, 512), prev_h (B, 512, 1), prev_c (B, 512, 1)

        y = v_center

        if prev_w is not None and prev_v is not None:
            y  = torch.cat([prev_w, v_center + prev_v], 1)

        y, (h1, c1) = self.lstm1(y, (prev_h1, prev_c1) if prev_h1 is not None else None) # (B, K + 1, 512), ((1, B, 512), (1, B, 512))

        h = h1.transpose(0, 1)   # (B, 1, 512)

        f_cross = torch.tanh(self.fc(h) + self.fc(v))   # (B, K + 1, 512)

        alpha = self.alpha_fc(f_cross).softmax(dim=1) # (B, K + 1, 1)

        new_v = alpha.transpose(1, 2) @ v # (B, 1, 512)

        y = torch.cat([new_v, h], 1)  # (B, 2, 512)

        y, (h2, c2) = self.lstm2(y, (prev_h2, prev_c2) if prev_h2 is not None else None) # (B, 2, 512), ((1, B, 512), (1, B, 512))

        logits = self.fc(h2.transpose(0, 1)) # (B, 1, 512)

        w = self.out(logits.squeeze(1)) # (B, VocabSize)

        return w, logits, new_v, h1, c1, h2, c2