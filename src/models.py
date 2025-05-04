import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, d_model, num_classes):
        super(Classifier, self).__init__()
        self.d_model = d_model
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, emb_features):
        x = emb_features['sentence_embedding']
        logits = self.classifier(x)
        return logits
    
class Transformer_Classifier(nn.Module):
    def __init__(self, sent_transformer, classifier):
        super(Transformer_Classifier, self).__init__()
        self.sent_transformer = sent_transformer
        self.classifier = classifier
        
    def forward(self, x):
        emb_features = self.sent_transformer(x)
        logits = self.classifier(emb_features)
        return logits