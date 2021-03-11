import random
import math
import numpy as np
from numpy.random import randint
from copy import copy
import pdb

class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalEndCrop(object):
    """Temporally crop the given frame indices at a beginning.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[-self.size:]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out







class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        # begin_index = 32
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalDense_train(object):
    """Dense Sampling from each video segment
    """
    def __init__(self, size, num_segments):
        self.size = size    
        self.num_segments = num_segments
    def __call__(self, frame_indices):
        """
        :param record: VideoRecord
        :return: list
        """
        # i3d dense sample
        t_stride = 64 // self.num_segments
        # t_stride = 128 // self.num_segments
        sample_pos = max(
            1, 1 + len(frame_indices) - t_stride * self.num_segments)
        start_idx = 0 if sample_pos == 1 else np.random.randint(
            0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx) % len(frame_indices)
                    for idx in range(self.num_segments)]
        # return np.array(offsets) + 1
        return np.array(offsets) 


class TemporalDense_test(object):
    """Dense Sampling from each video segment
    """
    def __init__(self, size, num_segments, clip_num):
        self.size = size    
        self.num_segments = num_segments
        self.clip_num = clip_num
    def __call__(self, frame_indices):
        """
        :param record: VideoRecord
        :return: list
        """
        # i3d dense sample
        t_stride = 64 // self.num_segments
        # t_stride = 128 // self.num_segments
        sample_pos = max(
            1, 1 + len(frame_indices) - t_stride * self.num_segments)
        # if num_clips == 1:
        #     start_idx = sample_pos // 2
        #     offsets = [(idx * t_stride + start_idx) % record.num_frames
        #                 for idx in range(self.num_segments)]
        # else:
        start_list = np.linspace(0,
                                sample_pos - 1,
                                num=self.clip_num,
                                dtype=int)
        offsets = []
        for start_idx in start_list.tolist():
            offsets += [
                (idx * t_stride + start_idx) % len(frame_indices)
                for idx in range(self.num_segments)
                    ]
        # pdb.set_trace()
        # return np.array(offsets) + 1
        return np.array(offsets).reshape(self.clip_num, -1) 



class TemporalUniformCrop_ego_train(object):
    """Random Sampling from each video segment
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        average_duration = len(frame_indices) // self.size
        if average_duration > 0:
            out = np.multiply(list(range(self.size)), average_duration) + randint(average_duration, size=self.size)
        else:              
            out = np.zeros((self.size,)).astype(np.int) + randint(len(frame_indices))
        return out    


class TemporalUniformCrop_ego_val(object):
    """Sampling for validation set
    Sample the middle frame from each video segment
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        average_duration = len(frame_indices) // self.size
        if len(frame_indices) > self.size:
            tick = len(frame_indices) / float(self.size)
            out = np.array([int(tick / 2.0 + tick * x) for x in range(self.size)])
        else:              
            out = np.zeros((self.size,)).astype(np.int) + int(len(frame_indices) // 2)
        return out 



class TemporalUniformCrop_train(object):
    """Random Sampling from each video segment
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        average_duration = len(frame_indices) // self.size
        if average_duration > 0:
            out = np.multiply(list(range(self.size)), average_duration) + randint(average_duration, size=self.size)
        else:                 
            out = np.zeros((self.size,)).astype(np.int)
        return out   




class TemporalUniformCrop_val(object):
    """Sampling for validation set
    Sample the middle frame from each video segment
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        average_duration = len(frame_indices) // self.size
        if len(frame_indices) > self.size:
            tick = len(frame_indices) / float(self.size)
            out = np.array([int(tick / 2.0 + tick * x) for x in range(self.size)])
        else:              
            out = np.zeros((self.size,)).astype(np.int)
        return out 


# for single clip test, if multiple clip test, use TemporalUniformCrop_train
class TemporalUniformCrop_test(object):
    """Sampling for test set
    Sample the middle frame from each video segment
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        tick = len(frame_indices) / float(self.size)
        out = np.array([int(tick / 2.0 + tick * x) for x in range(self.size)])
        return out 

# for multiple clip test
class TemporalUniform_test(object):
    """Sampling for test set
    Sample the middle frame from each video segment
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        tick = len(frame_indices) / float(self.size)
        out = np.array([int(tick / 2.0 + tick * x) for x in range(self.size)])
        return out 

# tick = (record.num_frames - self.new_length + 1) / float(
#                     self.num_segments)
#                 start_list = np.linspace(0, tick - 1, num=num_clips, dtype=int)
#                 offsets = []
#                 # print(start_list.tolist())
#                 # print(tick)
#                 for start_idx in start_list.tolist():
#                     offsets += [
#                         int(start_idx + tick * x) % record.num_frames
#                         for x in range(self.num_segments)
#                     ]

#             return np.array(offsets) + 1





# clip_len = 8
# num_frames = 3
# x = [i for i in range(num_frames)]
# t_val = TemporalUniformCrop_val(clip_len)
# t_train = TemporalUniformCrop_train(clip_len)
# t_test = TemporalUniformCrop_test(clip_len)
# print(x)
# print(t_train(copy(x)))
# print(t_val(copy(x)))
# print(t_test(copy(x)))
# pdb.set_trace()



class TemporalUniformCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        average_duration = len(frame_indices) // self.size
        if average_duration > 0:
            out = np.multiply(list(range(self.size)), average_duration) + randint(average_duration, size=self.size)
        else:
            out = frame_indices
            for index in out:
                if len(out) >= self.size:
                    break
                out.append(index)              
        return out






# class TemporalUniformCrop(object):
#     """Temporally crop the given frame indices at a center.
#     If the number of frames is less than the size,
#     loop the indices as many times as necessary to satisfy the size.
#     Args:
#         size (int): Desired output size of the crop.
#     """

#     def __init__(self,  skip, size):
#         self.skip = skip
#         self.size = size
#     def __call__(self, frame_indices):
#         """
#         Args:
#             frame_indices (list): frame indices to be cropped.
#         Returns:
#             list: Cropped frame indices.
#         """
#         clips = [frame_indices[i] for i in range(0, len(frame_indices), self.skip)]
#         out_clips = []
#         for clip_i_begin in clips:
#             begin_index = clip_i_begin
#             end_index = min(begin_index + self.size, len(frame_indices))
#             out = frame_indices[begin_index:end_index]
#             for index in out:
#                 if len(out) >= self.size:
#                     break
#                 out.append(index)
#             out_clips.append(out)
#             if begin_index + self.size >= len(frame_indices):
#                 break
#         # for index in out:
#         #     if len(out) >= self.size:
#         #         break
#         #     out.append(index)
#         return out_clips


# class TemporalUniformCrop(object):
#     """Temporally crop the given frame indices at a center.
#     If the number of frames is less than the size,
#     loop the indices as many times as necessary to satisfy the size.
#     Args:
#         size (int): Desired output size of the crop.
#     """

#     def __init__(self,  skip):
#         self.skip = skip
#     def __call__(self, frame_indices):
#         """
#         Args:
#             frame_indices (list): frame indices to be cropped.
#         Returns:
#             list: Cropped frame indices.
#         """
#         out = [frame_indices[i] for i in range(0, len(frame_indices), self.skip)]


#         return out