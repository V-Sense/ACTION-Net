from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import pdb


__all__ = ['BNInception', 'bninception']

pretrained_settings = {
    'bninception': {
        'imagenet': {
            'url': 'https://www.dropbox.com/s/3cvod6kzwluijcw/BNInception-9baff57459f5a1744.pth?dl=1',
            'input_space': 'BGR',
            'input_size': 224,
            'input_range': [0, 255],
            'mean': [104, 117, 128],
            'std': [1, 1, 1],
            'num_classes': 1000
        },
        'kinetics': {
            'url': 'https://www.dropbox.com/s/gx4u7itoyygix0c/BNInceptionKinetics-47f0695e.pth?dl=1',
            'input_space': 'BGR',
            'input_size': 224,
            'input_range': [0, 255],
            'mean': [104, 117, 128],  # [96.29023126, 103.16065604, 110.63666788]
            'std': [1, 1, 1],  # [40.02898126, 37.88248729, 38.7568578],
            'num_classes': 400
        }
    },
}


class Action(nn.Module):
    def __init__(self, net, n_segment=3, shift_div=8):
        super(Action, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.in_channels = self.net.in_channels
        self.out_channels = self.net.out_channels
        self.kernel_size = self.net.kernel_size
        self.stride = self.net.stride
        self.padding = self.net.padding
        self.reduced_channels = self.in_channels//16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // shift_div

        # shifting
        self.action_shift = nn.Conv1d(
                                    self.in_channels, self.in_channels,
                                    kernel_size=3, padding=1, groups=self.in_channels,
                                    bias=False)      
        self.action_shift.weight.requires_grad = True
        self.action_shift.weight.data.zero_()
        self.action_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
        self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right  


        if 2*self.fold < self.in_channels:
            self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed


        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), 
                                    stride=(1, 1 ,1), bias=False, padding=(1, 1, 1))       


        # # channel excitation
        self.action_p2_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, 
                                       groups=1)
        self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
    


        # motion excitation
        self.pad = (0,0,0,0,0,0,0,1)
        self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3), 
                                    stride=(1 ,1), bias=False, padding=(1, 1), groups=self.reduced_channels)
        self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        print('=> Using STME')


    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x_shift = x.view(n_batch, self.n_segment, c, h, w)
        x_shift = x_shift.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x_shift = x_shift.contiguous().view(n_batch*h*w, c, self.n_segment)
        x_shift = self.action_shift(x_shift)  # (n_batch*h*w, c, n_segment)
        x_shift = x_shift.view(n_batch, h, w, c, self.n_segment)
        x_shift = x_shift.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x_shift = x_shift.contiguous().view(nt, c, h, w)


        # 3D convolution: c*T*h*w, spatial temporal excitation
        nt, c, h, w = x_shift.size()
        x_p1 = x_shift.view(n_batch, self.n_segment, c, h, w).transpose(2,1).contiguous()
        x_p1 = x_p1.mean(1, keepdim=True)
        x_p1 = self.action_p1_conv1(x_p1)
        x_p1 = x_p1.transpose(2,1).contiguous().view(nt, 1, h, w)
        x_p1 = self.sigmoid(x_p1)
        x_p1 = x_shift * x_p1 + x_shift


        # 2D convolution: c*T*1*1, channel excitation
        x_p2 = self.avg_pool(x_shift)
        x_p2 = self.action_p2_squeeze(x_p2)
        nt, c, h, w = x_p2.size()
        x_p2 = x_p2.view(n_batch, self.n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
        x_p2 = self.action_p2_conv1(x_p2)
        x_p2 = self.relu(x_p2)
        x_p2 = x_p2.transpose(2,1).contiguous().view(-1, c, 1, 1)
        x_p2 = self.action_p2_expand(x_p2)
        x_p2 = self.sigmoid(x_p2)
        x_p2 = x_shift * x_p2 + x_shift
        

        # # 2D convolution: motion excitation
        x3 = self.action_p3_squeeze(x_shift)
        x3 = self.action_p3_bn1(x3)
        nt, c, h, w = x3.size()
        x3_plus0, _ = x3.view(n_batch, self.n_segment, c, h, w).split([self.n_segment-1, 1], dim=1)
        x3_plus1 = self.action_p3_conv1(x3)

        _ , x3_plus1 = x3_plus1.view(n_batch, self.n_segment, c, h, w).split([1, self.n_segment-1], dim=1)
        x_p3 = x3_plus1 - x3_plus0
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)
        x_p3 = self.avg_pool(x_p3.view(nt, c, h, w))
        x_p3 = self.action_p3_expand(x_p3)
        x_p3 = self.sigmoid(x_p3)
        x_p3 = x_shift * x_p3 + x_shift

        out = x_p1 + x_p2 + x_p3
        return out






class BNInception(nn.Module):
    def __init__(self, num_classes=1000, n_segment=8, fold_div=8):
        super(BNInception, self).__init__()
        inplace = True
        self._build_features(inplace, num_classes)
        # stage 2
        self.shift_2 = Action(self.inception_3a_1x1, n_segment, fold_div)
        # stage 3
        self.shift_3a = Action(self.inception_3b_1x1, n_segment, fold_div)
        self.shift_3b = Action(self.inception_3c_3x3_reduce, n_segment, fold_div)
        self.shift_3c = Action(self.inception_4a_1x1, n_segment, fold_div)

        self.shift_4a = Action(self.inception_4b_1x1, n_segment, fold_div)
        self.shift_4b = Action(self.inception_4c_1x1, n_segment, fold_div)
        self.shift_4c = Action(self.inception_4d_1x1, n_segment, fold_div)
                                                
        self.shift_4d = Action(self.inception_4e_3x3_reduce, n_segment, fold_div)
        self.shift_4e = Action(self.inception_5a_1x1, n_segment, fold_div)
        
        self.shift_5a = Action(self.inception_5b_1x1, n_segment, fold_div)


    def forward(self, x):
        # if self.input_space == 'BGR':
        #     assert len(x.size()) == 4
        #     x = x[:, (2, 1, 0)]
        x = self.features(x)
        x = self.logits(x)
        return x

    def features(self, x):
        # stage1
        # pdb.set_trace()
        pool1_3x3_s2_out = self._block_1(x)
        # stage2
        pool2_3x3_s2_out = self.shift_2(self._block_2(pool1_3x3_s2_out))
        # stage3
        inception_3a_output_out = self.shift_3a(self._block_3a(pool2_3x3_s2_out))
        inception_3b_output_out = self.shift_3b(self._block_3b(inception_3a_output_out))
        inception_3c_output_out = self.shift_3c(self._block_3c(inception_3b_output_out))


        inception_4a_output_out = self.shift_4a(self._block_4a(inception_3c_output_out))
        inception_4b_output_out = self.shift_4b(self._block_4b(inception_4a_output_out))
        inception_4c_output_out = self.shift_4c(self._block_4c(inception_4b_output_out))
        inception_4d_output_out = self.shift_4d(self._block_4d(inception_4c_output_out))
        inception_4e_output_out = self.shift_4e(self._block_4e(inception_4d_output_out))


        inception_5a_output_out = self.shift_5a(self._block_5a(inception_4e_output_out))
        inception_5b_output_out = self._block_5b(inception_5a_output_out)


        return inception_5b_output_out

    def logits(self, features):
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def build_temporal_ops(self, n_segment, is_temporal_shift='0' * 12, shift_div=8):
        # must call after loading weights
        self.n_segment = n_segment
        self.residual = 'res' in is_temporal_shift
        if self.residual:
            print('=> Using residual shift functions...')
        if is_temporal_shift in ['block', 'blockres']:
            self.is_temporal_shift = '1' * 12
        else:
            self.is_temporal_shift = is_temporal_shift
        self.is_temporal_shift = '0' + self.is_temporal_shift[1:]  # image input does not shift

        assert len(self.is_temporal_shift) == 12

        print('=> Injecting temporal shift with mask {}'.format(self.is_temporal_shift))
        self.fold_div = shift_div
        print('=> Using fold div: {}'.format(self.fold_div))

    def _temporal_forward_wrap(self, layer_func, index):
        if hasattr(self, 'is_temporal_shift') and self.is_temporal_shift[index] == '1':  # run temporal shuffling
            def wrapped_func(x, is_residual, n_segment, fold_div):
                if is_residual:
                    pdb.set_trace()
                    x_shift = Action(layer_func, n_segment, fold_div)
                    return F.relu(x + layer_func(x_shift))
                else:
                    pdb.set_trace()
                    x = TemporalShift.shift(x, n_segment, fold_div=fold_div)
                    return layer_func(x)
            from functools import partial
            return partial(wrapped_func, is_residual=self.residual, n_segment=self.n_segment,
                           fold_div=self.fold_div)
        else:
            return layer_func

    def _block_1(self, x):
        conv1_7x7_s2_out = self.conv1_7x7_s2(x)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_7x7_s2_bn_out)
        return pool1_3x3_s2_out

    def _block_2(self, x):
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(x)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_3x3_reduce_bn_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_3x3_bn_out)
        return pool2_3x3_s2_out

    def _block_3a(self, pool2_3x3_s2_out):
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(
            inception_3a_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = self.inception_3a_relu_double_3x3_reduce(
            inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = self.inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = self.inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = torch.cat(
            [inception_3a_1x1_bn_out, inception_3a_3x3_bn_out, inception_3a_double_3x3_2_bn_out,
             inception_3a_pool_proj_bn_out], 1)
        return inception_3a_output_out

    def _block_3b(self, inception_3a_output_out):
        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(inception_3a_output_out)
        inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(
            inception_3b_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = self.inception_3b_relu_double_3x3_reduce(
            inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = self.inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = self.inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = torch.cat(
            [inception_3b_1x1_bn_out, inception_3b_3x3_bn_out, inception_3b_double_3x3_2_bn_out,
             inception_3b_pool_proj_bn_out], 1)
        return inception_3b_output_out

    def _block_3c(self, inception_3b_output_out):
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = self.inception_3c_relu_3x3_reduce(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.inception_3c_3x3(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = self.inception_3c_relu_3x3(inception_3c_3x3_bn_out)
        inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(inception_3b_output_out)
        inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(
            inception_3c_double_3x3_reduce_out)
        inception_3c_relu_double_3x3_reduce_out = self.inception_3c_relu_double_3x3_reduce(
            inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = self.inception_3c_relu_double_3x3_1(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = self.inception_3c_relu_double_3x3_2(inception_3c_double_3x3_2_bn_out)
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = torch.cat(
            [inception_3c_3x3_bn_out, inception_3c_double_3x3_2_bn_out, inception_3c_pool_out], 1)
        return inception_3c_output_out

    def _block_4a(self, inception_3c_output_out):
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = self.inception_4a_relu_1x1(inception_4a_1x1_bn_out)
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = self.inception_4a_relu_3x3_reduce(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_out = self.inception_4a_3x3(inception_4a_3x3_reduce_bn_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = self.inception_4a_relu_3x3(inception_4a_3x3_bn_out)
        inception_4a_double_3x3_reduce_out = self.inception_4a_double_3x3_reduce(inception_3c_output_out)
        inception_4a_double_3x3_reduce_bn_out = self.inception_4a_double_3x3_reduce_bn(
            inception_4a_double_3x3_reduce_out)
        inception_4a_relu_double_3x3_reduce_out = self.inception_4a_relu_double_3x3_reduce(
            inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = self.inception_4a_relu_double_3x3_1(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = self.inception_4a_relu_double_3x3_2(inception_4a_double_3x3_2_bn_out)
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = self.inception_4a_relu_pool_proj(inception_4a_pool_proj_bn_out)
        inception_4a_output_out = torch.cat(
            [inception_4a_1x1_bn_out, inception_4a_3x3_bn_out, inception_4a_double_3x3_2_bn_out,
             inception_4a_pool_proj_bn_out], 1)
        return inception_4a_output_out

    def _block_4b(self, inception_4a_output_out):
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = self.inception_4b_relu_1x1(inception_4b_1x1_bn_out)
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = self.inception_4b_relu_3x3_reduce(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_out = self.inception_4b_3x3(inception_4b_3x3_reduce_bn_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = self.inception_4b_relu_3x3(inception_4b_3x3_bn_out)
        inception_4b_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(inception_4a_output_out)
        inception_4b_double_3x3_reduce_bn_out = self.inception_4b_double_3x3_reduce_bn(
            inception_4b_double_3x3_reduce_out)
        inception_4b_relu_double_3x3_reduce_out = self.inception_4b_relu_double_3x3_reduce(
            inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = self.inception_4b_relu_double_3x3_1(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = self.inception_4b_relu_double_3x3_2(inception_4b_double_3x3_2_bn_out)
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = self.inception_4b_relu_pool_proj(inception_4b_pool_proj_bn_out)
        inception_4b_output_out = torch.cat(
            [inception_4b_1x1_bn_out, inception_4b_3x3_bn_out, inception_4b_double_3x3_2_bn_out,
             inception_4b_pool_proj_bn_out], 1)
        return inception_4b_output_out

    def _block_4c(self, inception_4b_output_out):
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = self.inception_4c_relu_1x1(inception_4c_1x1_bn_out)
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = self.inception_4c_relu_3x3_reduce(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_out = self.inception_4c_3x3(inception_4c_3x3_reduce_bn_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = self.inception_4c_relu_3x3(inception_4c_3x3_bn_out)
        inception_4c_double_3x3_reduce_out = self.inception_4c_double_3x3_reduce(inception_4b_output_out)
        inception_4c_double_3x3_reduce_bn_out = self.inception_4c_double_3x3_reduce_bn(
            inception_4c_double_3x3_reduce_out)
        inception_4c_relu_double_3x3_reduce_out = self.inception_4c_relu_double_3x3_reduce(
            inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = self.inception_4c_relu_double_3x3_1(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = self.inception_4c_relu_double_3x3_2(inception_4c_double_3x3_2_bn_out)
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = self.inception_4c_relu_pool_proj(inception_4c_pool_proj_bn_out)
        inception_4c_output_out = torch.cat(
            [inception_4c_1x1_bn_out, inception_4c_3x3_bn_out, inception_4c_double_3x3_2_bn_out,
             inception_4c_pool_proj_bn_out], 1)
        return inception_4c_output_out

    def _block_4d(self, inception_4c_output_out):
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = self.inception_4d_relu_1x1(inception_4d_1x1_bn_out)
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = self.inception_4d_relu_3x3_reduce(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_out = self.inception_4d_3x3(inception_4d_3x3_reduce_bn_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = self.inception_4d_relu_3x3(inception_4d_3x3_bn_out)
        inception_4d_double_3x3_reduce_out = self.inception_4d_double_3x3_reduce(inception_4c_output_out)
        inception_4d_double_3x3_reduce_bn_out = self.inception_4d_double_3x3_reduce_bn(
            inception_4d_double_3x3_reduce_out)
        inception_4d_relu_double_3x3_reduce_out = self.inception_4d_relu_double_3x3_reduce(
            inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = self.inception_4d_relu_double_3x3_1(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = self.inception_4d_relu_double_3x3_2(inception_4d_double_3x3_2_bn_out)
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = self.inception_4d_relu_pool_proj(inception_4d_pool_proj_bn_out)
        inception_4d_output_out = torch.cat(
            [inception_4d_1x1_bn_out, inception_4d_3x3_bn_out, inception_4d_double_3x3_2_bn_out,
             inception_4d_pool_proj_bn_out], 1)
        return inception_4d_output_out

    def _block_4e(self, inception_4d_output_out):
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = self.inception_4e_relu_3x3_reduce(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_out = self.inception_4e_3x3(inception_4e_3x3_reduce_bn_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = self.inception_4e_relu_3x3(inception_4e_3x3_bn_out)
        inception_4e_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(inception_4d_output_out)
        inception_4e_double_3x3_reduce_bn_out = self.inception_4e_double_3x3_reduce_bn(
            inception_4e_double_3x3_reduce_out)
        inception_4e_relu_double_3x3_reduce_out = self.inception_4e_relu_double_3x3_reduce(
            inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = self.inception_4e_relu_double_3x3_1(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = self.inception_4e_relu_double_3x3_2(inception_4e_double_3x3_2_bn_out)
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        inception_4e_output_out = torch.cat(
            [inception_4e_3x3_bn_out, inception_4e_double_3x3_2_bn_out, inception_4e_pool_out], 1)
        return inception_4e_output_out

    def _block_5a(self, inception_4e_output_out):
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = self.inception_5a_relu_1x1(inception_5a_1x1_bn_out)
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = self.inception_5a_relu_3x3_reduce(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_out = self.inception_5a_3x3(inception_5a_3x3_reduce_bn_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = self.inception_5a_relu_3x3(inception_5a_3x3_bn_out)
        inception_5a_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(inception_4e_output_out)
        inception_5a_double_3x3_reduce_bn_out = self.inception_5a_double_3x3_reduce_bn(
            inception_5a_double_3x3_reduce_out)
        inception_5a_relu_double_3x3_reduce_out = self.inception_5a_relu_double_3x3_reduce(
            inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = self.inception_5a_relu_double_3x3_1(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = self.inception_5a_relu_double_3x3_2(inception_5a_double_3x3_2_bn_out)
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = self.inception_5a_relu_pool_proj(inception_5a_pool_proj_bn_out)
        inception_5a_output_out = torch.cat(
            [inception_5a_1x1_bn_out, inception_5a_3x3_bn_out, inception_5a_double_3x3_2_bn_out,
             inception_5a_pool_proj_bn_out], 1)
        return inception_5a_output_out

    def _block_5b(self, inception_5a_output_out):
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = self.inception_5b_relu_1x1(inception_5b_1x1_bn_out)
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = self.inception_5b_relu_3x3_reduce(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_out = self.inception_5b_3x3(inception_5b_3x3_reduce_bn_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = self.inception_5b_relu_3x3(inception_5b_3x3_bn_out)
        inception_5b_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(inception_5a_output_out)
        inception_5b_double_3x3_reduce_bn_out = self.inception_5b_double_3x3_reduce_bn(
            inception_5b_double_3x3_reduce_out)
        inception_5b_relu_double_3x3_reduce_out = self.inception_5b_relu_double_3x3_reduce(
            inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = self.inception_5b_relu_double_3x3_1(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = self.inception_5b_relu_double_3x3_2(inception_5b_double_3x3_2_bn_out)
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = self.inception_5b_relu_pool_proj(inception_5b_pool_proj_bn_out)
        inception_5b_output_out = torch.cat(
            [inception_5b_1x1_bn_out, inception_5b_3x3_bn_out, inception_5b_double_3x3_2_bn_out,
             inception_5b_pool_proj_bn_out], 1)
        return inception_5b_output_out

    def _build_features(self, inplace, num_classes):
        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace)
        self.pool1_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace)
        self.pool2_3x3_s2 = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_3a_1x1 = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace)
        self.inception_3a_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_3x3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_reduce = nn.Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3a_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3a_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3a_pool_proj = nn.Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3b_1x1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace)
        self.inception_3b_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_reduce = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3b_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3b_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_3b_pool_proj = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_3c_3x3_reduce = nn.Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_3x3 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_reduce = nn.Conv2d(320, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_3c_double_3x3_1 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_3c_double_3x3_2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_double_3x3_2_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_3c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_3c_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_4a_1x1 = nn.Conv2d(576, 224, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_1x1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_1x1 = nn.ReLU(inplace)
        self.inception_4a_3x3_reduce = nn.Conv2d(576, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3_reduce_bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_3x3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_3x3_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_3x3 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4a_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4a_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4a_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4b_1x1 = nn.Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_1x1 = nn.ReLU(inplace)
        self.inception_4b_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_3x3 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_3x3_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_3x3 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_reduce = nn.Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_double_3x3_reduce_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4b_double_3x3_1 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_1_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4b_double_3x3_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_2_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4b_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4b_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4b_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4c_1x1 = nn.Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_1x1 = nn.ReLU(inplace)
        self.inception_4c_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_3x3 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_3x3_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_3x3 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_reduce = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_double_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4c_double_3x3_1 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_1_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4c_double_3x3_2 = nn.Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_2_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4c_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4c_pool_proj = nn.Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4c_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4c_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4d_1x1 = nn.Conv2d(608, 96, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1_bn = nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_1x1 = nn.ReLU(inplace)
        self.inception_4d_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_3x3 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_reduce = nn.Conv2d(608, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4d_double_3x3_1 = nn.Conv2d(160, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_1_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4d_double_3x3_2 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_2_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4d_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_4d_pool_proj = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4d_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4d_relu_pool_proj = nn.ReLU(inplace)
        self.inception_4e_3x3_reduce = nn.Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_3x3_reduce_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_3x3 = nn.Conv2d(128, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_3x3_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_3x3 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_reduce = nn.Conv2d(608, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_4e_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_4e_double_3x3_1 = nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4e_double_3x3_1_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_4e_double_3x3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_double_3x3_2_bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True)
        self.inception_4e_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_4e_pool = nn.MaxPool2d((3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.inception_5a_1x1 = nn.Conv2d(1056, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_1x1 = nn.ReLU(inplace)
        self.inception_5a_3x3_reduce = nn.Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_3x3 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_reduce = nn.Conv2d(1056, 160, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_double_3x3_reduce_bn = nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5a_double_3x3_1 = nn.Conv2d(160, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5a_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5a_pool = nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=True)
        self.inception_5a_pool_proj = nn.Conv2d(1056, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5a_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5a_relu_pool_proj = nn.ReLU(inplace)
        self.inception_5b_1x1 = nn.Conv2d(1024, 352, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1_bn = nn.BatchNorm2d(352, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_1x1 = nn.ReLU(inplace)
        self.inception_5b_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_3x3 = nn.Conv2d(192, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_bn = nn.BatchNorm2d(320, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_3x3 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_reduce = nn.Conv2d(1024, 192, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_double_3x3_reduce_bn = nn.BatchNorm2d(192, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_reduce = nn.ReLU(inplace)
        self.inception_5b_double_3x3_1 = nn.Conv2d(192, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_1_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_1 = nn.ReLU(inplace)
        self.inception_5b_double_3x3_2 = nn.Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_2_bn = nn.BatchNorm2d(224, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_double_3x3_2 = nn.ReLU(inplace)
        self.inception_5b_pool = nn.MaxPool2d((3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), ceil_mode=True)
        self.inception_5b_pool_proj = nn.Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
        self.inception_5b_pool_proj_bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True)
        self.inception_5b_relu_pool_proj = nn.ReLU(inplace)
        self.global_pool = nn.AvgPool2d(7, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.fc = nn.Linear(1024, num_classes)


def bninception(pretrained='imagenet', n_segment=8, fold_div=8):
    r"""BNInception model architecture from <https://arxiv.org/pdf/1502.03167.pdf>`_ paper.
    """
    if pretrained is not None:
        print('=> Loading from pretrained model: {}'.format(pretrained))
        settings = pretrained_settings['bninception'][pretrained]
        num_classes = settings['num_classes']
        model = BNInception(num_classes=num_classes, n_segment=8, fold_div=8)
        model_dict = model.state_dict()
        model_dict.update(model_zoo.load_url(settings['url']))
        model.load_state_dict(model_dict)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        raise NotImplementedError
    return model


if __name__ == '__main__':
    model = bninception()
