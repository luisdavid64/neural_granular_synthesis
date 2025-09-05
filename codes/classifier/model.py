
import torch.nn as nn
import lightning as L
import torch


class WaveletClassifier(L.LightningModule):
    def __init__(self, w_model):
        super(WaveletClassifier, self).__init__()
        self.w_model = w_model
        for param in self.w_model.parameters():
            param.requires_grad = False
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifiers = None

    def forward(self, x):
        e = self.w_model.encode(x)
        print(e.shape)

    def training_step(self, batch):
        audio, sr, labels = batch
        with torch.no_grad():
            e = self.w_model.encode(audio)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.classier.parameters(), lr=0.005)