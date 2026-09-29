from torch import nn


class ClassificationHeads(nn.Module):
    def __init__(self, dim, hidden_dim=768, reference_style=False):
        super().__init__()
        if reference_style:
            self.shared = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Linear(dim, dim))
            def head(classes):
                return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Linear(dim, classes))
        else:
            self.shared = nn.Identity()
            def head(classes):
                return nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes))
        self.foul, self.severity = head(8), head(4)

    def forward(self, features):
        features = self.shared(features)
        return {'foul_logits': self.foul(features), 'severity_logits': self.severity(features)}

    def forward_task_features(self, foul_features, severity_features):
        return {'foul_logits': self.foul(self.shared(foul_features)),
                'severity_logits': self.severity(self.shared(severity_features))}
