import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class LogisticModel(nn.Module):

    def __init__(self,
                 classes_number=2,
                 input_dim=512,
                 dropout_rate=0.1):
        super().__init__()
        self.classes_number = classes_number
        self.input_dim = input_dim
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.fc_layer_1 = nn.Linear(input_dim, input_dim)
        self.fc_layer_2 = nn.Linear(input_dim, input_dim)
        self.fc_layer_3 = nn.Linear(input_dim, classes_number)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.dropout_layer(x)
        x = F.relu(self.fc_layer_1(x), inplace=False)
        x_1 = self.fc_layer_2(x)
        x_2 = F.relu(x_1, inplace=False)
        x = self.fc_layer_3(x_2)
        # 针对分类问题
        if self.classes_number > 1:
            output = self.softmax(x)
            return {
                'predictions': torch.max(output, dim=1)[1],
                'logistics': x,
                'fusion_features': x.cpu().detach().numpy().tolist()
            }
        else:
            output = x.clone()
            output_shape = output.shape
            output = output.detach_().view((output_shape[0]))
            return {
                'predictions': output,
                'logistics': x.view((output_shape[0])),
                'fusion_features': x_1.cpu().detach().numpy().tolist()
            }
