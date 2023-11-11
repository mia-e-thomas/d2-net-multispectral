import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, finetune_feature_extraction=False, use_cuda=True, vgg16_init=True):
        super(DenseFeatureExtractionModule, self).__init__()

        if vgg16_init: 
            model = models.vgg16()
            vgg16_layers = [
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
                'pool1',
                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
                'pool2',
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
                'pool3',
                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
                'pool4',
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
                'pool5'
            ]
            conv4_3_idx = vgg16_layers.index('conv4_3')

            self.model = nn.Sequential(
                *list(model.features.children())[: conv4_3_idx + 1]
            )

            # Fix forward parameters
            for param in self.model.parameters():
                param.requires_grad = False
            if finetune_feature_extraction:
                # Unlock conv4_3
                for param in list(self.model.parameters())[-2 :]:
                    param.requires_grad = True


        else: 
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2, stride=1),
                nn.Conv2d(256, 512, 3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            )

        self.num_channels = 512

        if use_cuda:
            self.model = self.model.cuda()


    def forward(self, batch):
        # ADDED
        self.model.to(torch.double)
        output = self.model(batch)
        return output


class SoftDetectionModule(nn.Module):
    def __init__(self, soft_local_max_size=3):
        super(SoftDetectionModule, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch)

        # Get the maximum value for each batch item (whole image)
        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
        
        # 1) Divide each batch item by its maximum value
        # 2) Take the exponential of the semi-normalized values [0-1]
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
        # 1) Pad 'exp' at edges w/ constant (value 1 at edges!)
        # 2) Average pool 
        # 3) Sum
        sum_exp = (
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
                self.soft_local_max_size, stride=1
            )
        )
        local_max_score = exp / sum_exp

        # Take max in another direction
        depth_wise_max = torch.max(batch, dim=1)[0]
        depth_wise_max_score = batch / depth_wise_max.unsqueeze(1)

        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0]

        score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1)

        return score


class D2Net(nn.Module):
    def __init__(self, model_file=None, use_cuda=True, vgg16_init=True):
        super(D2Net, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(
            finetune_feature_extraction=True,
            use_cuda=use_cuda,
            vgg16_init=vgg16_init,
        )

        self.detection = SoftDetectionModule()

        if model_file is not None:
            if use_cuda:
                self.load_state_dict(torch.load(model_file)['model'])
            else:
                self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])

    def forward(self, batch):
        b = batch['image1'].size(0)

        dense_features = self.dense_feature_extraction(
            torch.cat([batch['image1'], batch['image2']], dim=0)
        )

        scores = self.detection(dense_features)

        dense_features1 = dense_features[: b, :, :, :]
        dense_features2 = dense_features[b :, :, :, :]

        scores1 = scores[: b, :, :]
        scores2 = scores[b :, :, :]

        return {
            'dense_features1': dense_features1,
            'scores1': scores1,
            'dense_features2': dense_features2,
            'scores2': scores2
        }
