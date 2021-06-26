import torch.nn as nn
import transformers


class SentimentalModel(nn.Module):
    def __init__(self):
        super(SentimentalModel, self).__init__()
        self.backbone = transformers.CamembertForSequenceClassification.from_pretrained("camembert-base")   

    def forward(self, input):
        output = self.backbone(input)
        return output
