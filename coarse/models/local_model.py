import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D convolution is used for the decoder. It acts as a standard FC, but allows to use a batch of point samples features,
# additionally to the batch over the input objects.
# The dimensions are used as follows:
# batch_size (N) = #3D objects , channels = features, signal_lengt (L) (convolution dimension) = #point samples
# kernel_size = 1 i.e. every convolution is done over only all features of one point sample, this makes it a FC.

# ShapeNet Pointcloud Completion ---------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

class AnimalsContour(nn.Module):

    def __init__(self, hidden_dim=256):
        super(AnimalsContour, self).__init__()
        # 128**3 res input
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='zeros')
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='zeros')
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='zeros')
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='zeros')
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)


        displacement = 0.0722
        displacements = []
        displacements.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacement
                displacements.append(input)
        # displacements [[0, 0, 0], [-0.0722, 0, 0], [0.0722, 0, 0], [0, -0.0722, 0], [0, 0.0722, 0], [0, 0, -0.0722], [0, 0, 0.0722]]
        self.displacements = torch.Tensor(displacements).cuda()

    def forward(self, p, x):
        x = x.unsqueeze(1)
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacements], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p, padding_mode='zeros', align_corners=True)  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='zeros', align_corners=True)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='zeros', align_corners=True)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='zeros', align_corners=True)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='zeros', align_corners=True)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='zeros', align_corners=True)

        # here every channel corresponds to one feature.
        # print("feature_0:", feature_0.shape)
        # print("feature_1:", feature_1.shape)
        # print("feature_2:", feature_2.shape)
        # print("feature_3:", feature_3.shape)
        # print("feature_4:", feature_4.shape)
        # print("feature_5:", feature_5.shape)

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)
        # print("features:", features.shape)
        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)
        # print("out:", out.shape)
        return out

