# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from ..builder import PIPELINES

EPS = 1e-3


@PIPELINES.register_module()
class GeneratePoseTarget:
    """Generate pseudo heatmaps based on joint coordinates and confidence.

    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".

    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
        left_limb (tuple[int]): Indexes of left limbs, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left limbs of skeletons we defined for COCO-17p.
        right_limb (tuple[int]): Indexes of right limbs, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right limbs of skeletons we defined for COCO-17p.
    """

    def __init__(self,
                 sigma=0.6,
                 use_score=True,
                 with_kp=True,
                 with_limb=False,
                 input_format='Heatmap',
                 noise=False,
                 model_type='poseconv',
                 skeletons=((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                            (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                            (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)),
                 double=False,
                 left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp=(2, 4, 6, 8, 10, 12, 14, 16),
                 left_limb=(0, 2, 4, 5, 6, 10, 11, 12),
                 right_limb=(1, 3, 7, 8, 9, 13, 14, 15),
                 scaling=1.):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double
        self.input_format = input_format

        assert self.with_kp + self.with_limb == 1, ('One of "with_limb" and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.left_limb = left_limb
        self.right_limb = right_limb
        self.scaling = scaling
        self.noise = noise
        self.model_type = model_type

    def generate_a_heatmap(self, arr, centers, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
            max_values (np.ndarray): The max values of each keypoint. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for center, max_value in zip(centers, max_values):
            if max_value < EPS:
                continue

            mu_x, mu_y = center[0], center[1]
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
            patch = patch * max_value
            arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

    def generate_a_limb_heatmap(self, arr, starts, ends, start_values, end_values):
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: M * 2.
            ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: M * 2.
            start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: M.
            end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for start, end, start_value, end_value in zip(starts, ends, start_values, end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < EPS:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0]) ** 2 + (y - start[1]) ** 2)

            # distance to end keypoints
            d2_end = ((x - end[0]) ** 2 + (y - end[1]) ** 2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

            if d2_ab < 1:
                self.generate_a_heatmap(arr, start[None], start_value[None])
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
            d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

            patch = np.exp(-d2_seg / 2. / sigma ** 2)
            patch = patch * value_coeff

            arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)

    def generate_heatmap(self, arr, kps, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: V * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame. Shape: M * V * 2.
            max_values (np.ndarray): The confidence score of each keypoint. Shape: M * V.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                self.generate_a_heatmap(arr[i], kps[:, i], max_values[:, i])

        if self.with_limb:
            for i, limb in enumerate(self.skeletons):
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                self.generate_a_limb_heatmap(arr[i], starts, ends, start_values, end_values)

    def add_random_noise(self, kps, h, w):
        mean = 0
        disp = 0.01 * (h + w) / 2

        kps_flat = kps.reshape(-1, 2)

        new_kps_flat = []
        for point in kps_flat:
            x, y = point

            delta_x, delta_y = np.random.normal(mean, disp, size=2)

            new_x = max(0, min(h, x + delta_x))
            new_y = max(0, min(w, y + delta_y))

            new_kps_flat.append([new_x, new_y])

        new_kps_flat = np.array(new_kps_flat)
        new_kps = new_kps_flat.reshape(kps.shape)
        return new_kps


    def plot_skeleton(self, keypoints, pairs, title):
        fig, ax = plt.subplots()

        scatter = ax.scatter([], [], color='red', label='Keypoints')
        lines = []
        def update(frame):
            ax.cla()

            ax.set_xlim([-10, 60])
            ax.set_ylim([-10, 60])

            scatter.set_offsets(keypoints[frame])

            for connection in pairs:
                start_point = keypoints[frame][connection[0]]
                end_point = keypoints[frame][connection[1]]
                line = ax.plot(*zip(start_point, end_point), color='blue')
                lines.append(line)

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(keypoints), interval=200)
        return ani

        # Set up the video writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)

        # Save the animation as a video file
        ani.save(f'videos/{title}.mp4', writer=writer)

    def gen_an_aug(self, results):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint']
        kp_shape = all_kps.shape

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']

        # scale img_h, img_w and kps
        img_h = int(img_h * self.scaling + 0.5)
        img_w = int(img_w * self.scaling + 0.5)
        all_kps[..., :2] *= self.scaling

        num_frame = kp_shape[1]
        num_c = 0
        if self.with_kp:
            num_c += all_kps.shape[2]
        if self.with_limb:
            num_c += len(self.skeletons)
        ret = np.zeros([num_frame, num_c, img_h, img_w], dtype=np.float32)

        for i in range(num_frame):
            # M, V, C
            kps = all_kps[:, i]
            # M, C
            kpscores = all_kpscores[:, i] if self.use_score else np.ones_like(all_kpscores[:, i])

            self.generate_heatmap(ret[i], kps, kpscores)

        if self.input_format == 'NCTHW_Heatmap':
            return ret

        elif self.input_format == 'Skeleton':
            if all_kps.shape[0] == 1:
                all_kps = np.concatenate([all_kps, all_kps], axis=0)
                all_kpscores = np.concatenate([all_kpscores, all_kpscores], axis=0)

            if self.model_type == 'motionbert':
                res = np.concatenate((all_kps[:2, :, :, :], all_kpscores[:2, :, :, None]), axis=-1)
                return res.astype(np.float32)

            return all_kps[:2].astype(np.float32)
            # return all_kps.astype(np.float32)

        elif self.input_format == 'Grayscale':
            return np.sum(ret, axis=1)

    def __call__(self, results):
        heatmap = self.gen_an_aug(results)
        key = 'heatmap_imgs' if 'imgs' in results else 'imgs'

        if self.double:
            indices = np.arange(heatmap.shape[1], dtype=np.int64)
            left, right = (self.left_kp, self.right_kp) if self.with_kp else (self.left_limb, self.right_limb)
            for l, r in zip(left, right):  # noqa: E741
                indices[l] = r
                indices[r] = l
            heatmap_flip = heatmap[..., ::-1][:, indices]
            heatmap = np.concatenate([heatmap, heatmap_flip])
        results[key] = heatmap
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str


# The Input will be a feature map ((N x T) x H x W x K), The output will be
# a 2D map: (N x H x W x [K * (2C + 1)])
# N is #clips x #crops, K is num_kpt
@PIPELINES.register_module()
class Heatmap2Potion:

    def __init__(self, C, option='full'):
        self.C = C
        self.option = option
        self.eps = 1e-4
        assert isinstance(C, int)
        assert C >= 2
        assert self.option in ['U', 'N', 'I', 'full']

    def __call__(self, results):
        heatmaps = results['imgs']

        if 'clip_len' in results:
            clip_len = results['clip_len']
        else:
            # Just for Video-PoTion generation
            clip_len = heatmaps.shape[0]

        C = self.C
        heatmaps = heatmaps.reshape((-1, clip_len) + heatmaps.shape[1:])
        # num_clip, clip_len, C, H, W
        heatmaps = heatmaps.transpose(0, 1, 3, 4, 2)

        # t in {0, 1, 2, ..., clip_len - 1}
        def idx2color(t):
            st = np.zeros(C, dtype=np.float32)
            ed = np.zeros(C, dtype=np.float32)
            if t == clip_len - 1:
                ed[C - 1] = 1.
                return ed
            val = t / (clip_len - 1) * (C - 1)
            bin_idx = int(val)
            val = val - bin_idx
            st[bin_idx] = 1.
            ed[bin_idx + 1] = 1.
            return (1 - val) * st + val * ed

        heatmaps_wcolor = []
        for i in range(clip_len):
            color = idx2color(i)
            heatmap = heatmaps[:, i]
            heatmap = heatmap[..., None]
            heatmap = np.matmul(heatmap, color[None,])
            heatmaps_wcolor.append(heatmap)

        # The shape of each element is N x H x W x K x C
        heatmap_S = np.sum(heatmaps_wcolor, axis=0)
        # The shape of U_norm is N x 1 x 1 x K x C
        U_norm = np.max(
            np.max(heatmap_S, axis=1, keepdims=True), axis=2, keepdims=True)
        heatmap_U = heatmap_S / (U_norm + self.eps)
        heatmap_I = np.sum(heatmap_U, axis=-1, keepdims=True)
        heatmap_N = heatmap_U / (heatmap_I + 1)
        if self.option == 'U':
            heatmap = heatmap_U
        elif self.option == 'I':
            heatmap = heatmap_I
        elif self.option == 'N':
            heatmap = heatmap_N
        elif self.option == 'full':
            heatmap = np.concatenate([heatmap_U, heatmap_I, heatmap_N],
                                     axis=-1)

        # Reshape the heatmap to 4D
        heatmap = heatmap.reshape(heatmap.shape[:3] + (-1,))
        results['imgs'] = heatmap
        return results
