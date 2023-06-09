# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
import random

import numpy as np
from matplotlib import animation, pyplot as plt
from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms, noise=False, noise_alpha=0.003):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        self.noise = noise
        self.noise_alpha = noise_alpha
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def add_random_noise(self, kps, h, w):
        kps = kps.astype(float)
        pairwise_diff = kps[0][:-1] - kps[0][1:]
        distances = np.linalg.norm(pairwise_diff, axis=2)
        total_difference = np.sum(distances)

        diff_coeff = max(0.2, min(1, np.sqrt(total_difference / 10000)))
        alpha = self.noise_alpha * diff_coeff

        mean = 0
        std_h = alpha * h
        std_w = alpha * w
        kps_flat = kps.reshape(-1, 2)

        new_kps_flat = []
        for point in kps_flat:
            x, y = point

            delta_x = np.random.normal(mean, std_w, size=1)[0]
            delta_y = np.random.normal(mean, std_h, size=1)[0]

            new_x = x + delta_x
            new_y = y + delta_y

            new_kps_flat.append([new_x, new_y])

        new_kps_flat = np.array(new_kps_flat)
        new_kps = new_kps_flat.reshape(kps.shape)
        return new_kps

    def plot_skeleton_sc(self, keypoints, pairs, h, w, title):
        x = keypoints[0, :, 0]
        y = keypoints[0, :, 1]

        # Plot the keypoints
        plt.figure()
        plt.scatter(x, y, c='r')

        for limb in pairs:
            plt.plot([x[limb[0]], x[limb[1]]], [y[limb[0]], y[limb[1]]], 'g-', linewidth=2)

        # Set axis limits
        plt.xlim(np.min(x) - 1, np.max(x) + 1)
        plt.ylim(np.max(y) + 1, np.min(y) - 1)

        plt.show()

    def plot_skeleton(self, init_kps, noise_kps, pairs, h, w, title):
        print('Title', title)
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # scatter1 = ax1.scatter([], [], s=5, color='red', label='Keypoints')
        # scatter2 = ax2.scatter([], [], s=5, color='red', label='Keypoints')

        minx, maxx = np.min(init_kps[:, :, 0]) - 50, np.max(init_kps[:, :, 0]) + 50
        miny, maxy = np.min(init_kps[:, :, 1]) - 50, np.max(init_kps[:, :, 1]) + 50

        def update(frame):
            ax1.cla()
            ax2.cla()

            scatter1 = ax1.scatter([], [], c='red', s=10, animated=True)
            scatter2 = ax2.scatter([], [], c='red', s=10, animated=True)

            ax1.set_xlim(minx, maxx)
            ax1.set_ylim(maxy, miny)
            ax2.set_xlim(minx, maxx)
            ax2.set_ylim(maxy, miny)

            ax1.axis('equal')
            ax2.axis('equal')

            scatter1.set_offsets(init_kps[frame])
            scatter2.set_offsets(noise_kps[frame])

            for connection in pairs:
                start_point1 = init_kps[frame][connection[0]]
                end_point1 = init_kps[frame][connection[1]]

                start_point2 = noise_kps[frame][connection[0]]
                end_point2 = noise_kps[frame][connection[1]]

                ax1.plot(*zip(start_point1, end_point1), color='blue', alpha=1)
                ax2.plot(*zip(start_point2, end_point2), color='blue', alpha=1)

            return scatter1, scatter2

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(init_kps), interval=200)

        # Set up the video writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'))

        ani.save(f'videos_compose/{title}.mp4', writer=writer)
        print('! videos_compose')
        # ani.save(f'videos_compose/animation_{title}.gif', writer=writer)


    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """

        skeletons = ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                     (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                     (13, 15), (6, 12), (12, 14), (14, 16), (11, 12))

        first_iter = True
        random_number = random.randint(0, 10000)

        for t in self.transforms:

            if first_iter and self.noise:
                first_iter = False
                h, w = data['img_shape']
                init_kps = data['keypoint'].copy()

                if self.noise_alpha != 0:
                    data['keypoint'] = self.add_random_noise(data['keypoint'], h, w)
                    label = data['label']

            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
