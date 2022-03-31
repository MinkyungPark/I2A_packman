import imp
import torch
import torch.nn as nn
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_shape, n1, n2, n3):
        super(BasicBlock, self).__init__()
        
        self.in_shape = in_shape
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        
        self.maxpool = nn.MaxPool2d(kernel_size=in_shape[1:])
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n1, kernel_size=1, stride=2, padding=6),
            nn.ReLU(),
            nn.Conv2d(n1, n1, kernel_size=10, stride=1, padding=(5, 6)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n2, n2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n1 + n2,  n3, kernel_size=1),
            nn.ReLU()
        )
        
    def forward(self, inputs):
        x = self.pool_and_inject(inputs)
        x = torch.cat([self.conv1(x), self.conv2(x)], 1)
        x = self.conv3(x)
        x = torch.cat([x, inputs], 1)
        return x
    
    def pool_and_inject(self, x):
        pooled = self.maxpool(x)
        tiled = pooled.expand((x.size(0),) + self.in_shape)
        out = torch.cat([tiled, x], 1)
        return out

  
class EnvModel(nn.Module):
    def __init__(self, in_shape, num_pixels, num_rewards):
        super(EnvModel, self).__init__()
        width = in_shape[1]
        height = in_shape[2]
        
        self.conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1),
            nn.ReLU()
        )
        
        self.basic_block1 = BasicBlock((64, width, height), 16, 32, 64)
        self.basic_block2 = BasicBlock((128, width, height), 16, 32, 64)
        
        self.image_conv = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1),
            nn.ReLU()
        )
        self.image_fc = nn.Linear(256, num_pixels)
        
        self.reward_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU()
        )
        self.reward_fc = nn.Linear(64 * width * height, num_rewards)
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        
        x = self.conv(inputs)
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        
        image = self.image_conv(x)
        image = image.permute(0, 2, 3, 1).contiguous().view(-1, 256)
        image = self.image_fc(image)

        reward = self.reward_conv(x)
        reward = reward.view(batch_size, -1)
        reward = self.reward_fc(reward)
        
        return image, reward


class ModelDyna():
    def __init__(self):
        self.pixels = (
            (0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0), 
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 0.0), 
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
        self.pixel_to_categorical = {pix:i for i, pix in enumerate(self.pixels)}
        self.num_pixels = len(self.pixels)

        self.mode_rewards = {
            "regular": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "avoid": [0.1, -0.1, -5, -10, -20],
            "hunt": [0, 1, 10, -20],
            "ambush": [0, -0.1, 10, -20],
            "rush": [0, -0.1, 9.9]
        }
        self.reward_to_categorical = {mode: {reward:i for i, reward in enumerate(self.mode_rewards[mode])} for mode in self.mode_rewards.keys()}

    def pix_to_target(self, next_states):
        target = []
        for pixel in next_states.transpose(0, 2, 3, 1).reshape(-1, 3):
            target.append(self.pixel_to_categorical[tuple([np.ceil(pixel[0]), np.ceil(pixel[1]), np.ceil(pixel[2])])])
        return target

    def target_to_pix(self, imagined_states):
        pixels = []
        to_pixel = {value: key for key, value in self.pixel_to_categorical.items()}
        for target in imagined_states:
            pixels.append(list(to_pixel[target]))
        return np.array(pixels)

    def rewards_to_target(self, mode, rewards):
        target = []
        for reward in rewards:
            target.append(self.reward_to_categorical[mode][reward])
        return target


