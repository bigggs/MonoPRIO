import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MonoPRIORouter(nn.Module):

    def __init__(self, prior_path: str, input_dim: int = 256, visual_dim: int = 1024):
        super().__init__()

        bank = np.load(prior_path, allow_pickle=True)

        bank_visual = torch.from_numpy(bank["visual"]).float()
        bank_mu = torch.from_numpy(bank["mu"]).float()
        bank_sigma = torch.from_numpy(bank["sigma"]).float()
        bank_mu_log = torch.from_numpy(bank["mu_log"]).float()
        bank_V_log = torch.from_numpy(bank["V_log"]).float()
        bank_inv_std_log = torch.from_numpy(bank["inv_std_log"]).float()

        k_clusters = bank_visual.shape[0]

        if "class_counts" in bank.files:
            class_counts = bank["class_counts"].astype(np.int64)
            offsets = np.concatenate([[0], np.cumsum(class_counts)])
            self.register_buffer("class_offsets", torch.from_numpy(offsets).long())
            self.num_classes = len(class_counts)
        else:
            self.class_offsets = None
            self.num_classes = None

        self.register_buffer("bank_visual", bank_visual)
        self.register_buffer("bank_mu", bank_mu)
        self.register_buffer("bank_sigma", bank_sigma)
        self.register_buffer("bank_mu_log", bank_mu_log)
        self.register_buffer("bank_V_log", bank_V_log)
        self.register_buffer("bank_inv_std_log", bank_inv_std_log)

        self.k_clusters = k_clusters
        self.q_proj = nn.Linear(input_dim, 256)
        self.k_proj = nn.Linear(visual_dim, 256)
        self.scale = 256 ** -0.5

    def forward(self, queries: torch.Tensor, class_probs: torch.Tensor = None):
        q = F.normalize(self.q_proj(queries), dim=-1)
        k = F.normalize(self.k_proj(self.bank_visual), dim=-1)
        logits = torch.matmul(q, k.t()) * self.scale
        eps = 1e-6

        if (class_probs is not None) and (self.class_offsets is not None):
            class_probs = class_probs / class_probs.sum(-1, keepdim=True).clamp_min(eps)
            attn_full = torch.zeros_like(logits)

            for c in range(self.num_classes):
                s = int(self.class_offsets[c].item())
                e = int(self.class_offsets[c + 1].item())
                if e <= s:
                    continue
                logits_c = logits[:, :, s:e]
                attn_c = F.softmax(logits_c, dim=-1)
                pc = class_probs[:, :, c].detach().unsqueeze(-1)
                attn_full[:, :, s:e] = attn_c * pc

            attn_weights = attn_full / attn_full.sum(-1, keepdim=True).clamp_min(eps)
        else:
            attn_weights = F.softmax(logits, dim=-1)

        prior_mu = torch.matmul(attn_weights, self.bank_mu)
        second_moments = self.bank_sigma ** 2 + self.bank_mu ** 2
        variance_term = torch.matmul(attn_weights, second_moments) - prior_mu ** 2
        prior_sigma = torch.sqrt(torch.clamp(variance_term, min=1e-6))

        return (
            prior_mu,
            prior_sigma,
            attn_weights,
            self.bank_mu_log,
            self.bank_V_log,
            self.bank_inv_std_log,
        )
