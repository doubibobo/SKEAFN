import torch


class EMA:

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay
                               ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM:
    """
    参考自: https://blog.csdn.net/qq_40176087/article/details/121512229
    FGSM的更新公式为: eplison * torch.sign(param.grad)
    """

    def __init__(self, model, eps=1.) -> None:
        self.model = model
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, embedding_name="emb"):
        # for name, param in self.model.named_parameters():
        #     print(name)
        for name, param in self.model.named_parameters():
            if param.requires_grad and embedding_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, embedding_name="emb"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embedding_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
