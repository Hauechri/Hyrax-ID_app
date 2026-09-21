import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# TDNN Layer
# -------------------------
class TDNNLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 context_size=5,
                 dilation=1):
        super().__init__()

        padding = ((context_size - 1) * dilation) // 2

        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=context_size,
            dilation=dilation,
            padding=padding
        )

    def forward(self, x):

        x = self.conv(x)

        x = F.relu(x)

        return x


# -------------------------
# Element Encoder (TDNN)
# -------------------------
class TDNNElementEncoder(nn.Module):
    def __init__(self,
                 input_dim=64,
                 emb_dim=128):
        super().__init__()

        self.tdnn1 = TDNNLayer(
            input_dim,
            256,
            context_size=5
        )

        self.tdnn2 = TDNNLayer(
            256,
            256,
            context_size=3,
            dilation=2
        )

        self.tdnn3 = TDNNLayer(
            256,
            256,
            context_size=3,
            dilation=3
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x):
        """
        x: (T, F)
        """

        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, T, F)

        # ONLY TRANSPOSE ONCE
        x = x.transpose(1, 2)  # (1, F, T)

        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return x.squeeze(0)


# -------------------------
# Full Model
# -------------------------
class TDNNBoutModel(nn.Module):
    def __init__(self,
                 num_classes,
                 input_dim=64,
                 emb_dim=128):
        super().__init__()

        self.encoder = TDNNElementEncoder(
            input_dim=input_dim,
            emb_dim=emb_dim
        )

        self.classifier = nn.Linear(
            emb_dim,
            num_classes
        )

    def forward(self, batch_elements):

        batch_embeddings = []

        for elements in batch_elements:

            elem_embs = []

            for e in elements:

                emb = self.encoder(e)

                elem_embs.append(emb)

            elem_embs = torch.stack(elem_embs)

            bout_emb = elem_embs.mean(dim=0)

            batch_embeddings.append(bout_emb)

        batch_embeddings = torch.stack(batch_embeddings)

        logits = self.classifier(batch_embeddings)

        return logits