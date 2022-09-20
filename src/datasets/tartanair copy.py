import os
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from .transformer import Transformer
from .transformation import pos_quats2SE_matrices
from .trajectory_transform import ned2cam


class TartanAirLoader(object):
    def __init__(
        self, tartanairDir, phase, downsample_ratio=0.5, disp_norm=False
    ):
        self.phase = phase
        self.downsample_ratio = downsample_ratio
        self.tartanairDir = tartanairDir
        self.files = []
        self.transformer = Transformer(phase).get_transform()
        self.disp_norm = disp_norm

        currpath = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(
            currpath, "filenames/{}_list.txt".format(self.phase)
        )
        extrpath = os.path.join(
            currpath, "filenames/{}_extrinsic.npy".format(self.phase)
        )

        self.extrinsics = np.load(extrpath)


        [
                    "depth_left",
                    "depth_right",
                    "image_left",
                    "image_right",
                    "seg_left",
                    "seg_right",
                ]
        with open(filepath, "r") as f:
            data_list = f.read().split("\n")
            for data in data_list:
                if len(data) == 0:
                    continue

                (
                    traj_path,
                    src_flag,
                    src_index,
                    tgt_flag,
                    tgt_index,
                ) = data.split(" ")
                src_side = "left" if src_flag == "l" else "right"
                tgt_side = "left" if tgt_flag == "l" else "right"
                path_fmt = "{}_{}/{:06d}_{}{}.{}"

                #Cria uma lista de arquivos -> gen_file_list()
                self.files.append(
                    {
                        "src_rgb": os.path.join(
                            traj_path,
                            path_fmt.format(
                                "image",
                                src_side,
                                int(src_index),
                                src_side,
                                "",
                                "png",
                            ),
                        ),
                        "tgt_rgb": os.path.join(
                            traj_path,
                            path_fmt.format(
                                "image",
                                tgt_side,
                                int(tgt_index),
                                tgt_side,
                                "",
                                "png",
                            ),
                        ),
                        "src_depth": os.path.join(
                            traj_path,
                            path_fmt.format(
                                "depth",
                                src_side,
                                int(src_index),
                                src_side,
                                "_depth",
                                "npy",
                            ),
                        ),
                        "tgt_depth": os.path.join(
                            traj_path,
                            path_fmt.format(
                                "depth",
                                tgt_side,
                                int(tgt_index),
                                tgt_side,
                                "_depth",
                                "npy",
                            ),
                        ),
                        "src_mask": os.path.join(
                            traj_path,
                            path_fmt.format(
                                "mask",
                                src_side,
                                int(src_index),
                                src_side,
                                "_mask_v2",
                                "png",
                            ),
                        ),
                        "tgt_mask": os.path.join(
                            traj_path,
                            path_fmt.format(
                                "mask",
                                tgt_side,
                                int(tgt_index),
                                tgt_side,
                                "_mask_v2",
                                "png",
                            ),
                        ),
                        "src_index": int(src_index),
                        "tgt_index": int(tgt_index),
                    }
                )

    def __len__(self):
        return len(self.files)

    def _check_path(self, filename, err_info):
        file_path = os.path.join(self.tartanairDir, filename)

        assert os.path.exists(file_path), err_info
        return file_path

    def _get_intrinsic(self, w, h):
        fx = 320.0 * (w / 640)  # focal length x
        fy = 320.0 * (w / 640)  # focal length y
        cx = 320.0 * (h / 480)  # optical center x
        cy = 240.0 * (h / 480)  # optical center y

        k_src, k_tgt = np.eye(4), np.eye(4)

        """
            |fx  0   cx  0 |
            |0   fy  cy  0 |
            |0   0   1   0 |
            |0   0   0   1 |
        """

        k_src[0, 0], k_src[1, 1], k_src[0, 2], k_src[1, 2] = fx, fy, cx, cy
        k_tgt[0, 0], k_tgt[1, 1], k_tgt[0, 2], k_tgt[1, 2] = fx, fy, cx, cy

        return k_src, k_tgt

    def _read_data(
        self,
        mode,
        item_files,
        index,
        downsample_ratio=0.5,
        outpainting=False,
        w=512,
        h=256,
        op_f=0.03,
    ):
        #    def _read_data(self, mode, item_files, index, downsample_ratio=0.5, outpainting=False, w=512, h=256, op_f=0.03):
        #    def _read_data(self, mode, item_files, index, downsample_ratio=0.5, w=640, h=480):
        #    def _read_data(self, mode, item_files, index, downsample_ratio=0.5, w=640, h=480):
        # print(item_files['src_rgb'])
        # print(item_files['tgt_rgb'])
        # print(item_files['src_depth'])
        # print(item_files['tgt_depth'])
        # print(item_files['src_mask'])
        src_rgb_path = self._check_path(
            item_files["src_rgb"], "Panic::Cannot find Left Image "
        )
        tgt_rgb_path = self._check_path(
            item_files["tgt_rgb"], "Panic::Cannot find Right Image"
        )
        src_depth_path = self._check_path(
            item_files["src_depth"], "Panic::Cannot find Left Depth"
        )
        tgt_depth_path = self._check_path(
            item_files["tgt_depth"], "Panic::Cannot find Right Depth"
        )
        src_mask_path = self._check_path(
            item_files["src_mask"], "Panic::Cannot find Left Mask"
        )
        tgt_mask_path = self._check_path(
            item_files["tgt_mask"], "Panic::Cannot find Right Mask"
        )

        pose_ind = 0  # src -> tgt

        invert_src_tgt = np.random.rand() > 0.5

        # get camera intrinsic matrix
        src_k, tgt_k = self._get_intrinsic(w, h)
        src_k = np.linalg.inv(src_k)

        src_idx, tgt_idx = "src_index", "tgt_index"
        if mode == "train" and invert_src_tgt:
            src_rgb_path, tgt_rgb_path = tgt_rgb_path, src_rgb_path
            src_depth_path, tgt_depth_path = tgt_depth_path, src_depth_path
            src_mask_path, tgt_mask_path = tgt_mask_path, src_mask_path
            pose_ind = 1  # tgt -> src
            src_idx, tgt_idx = tgt_idx, src_idx

        # load and reshape images
        src_rgb = Image.open(src_rgb_path)
        tgt_rgb = Image.open(tgt_rgb_path)
        src_mask = Image.open(src_mask_path).convert("L")

        src_rgb = src_rgb.resize((w, h), resample=Image.BICUBIC)
        tgt_rgb = tgt_rgb.resize((w, h), resample=Image.BICUBIC)
        src_mask = src_mask.resize((w, h), resample=Image.BICUBIC)

        src_rgb_outpainting = src_rgb.copy()
        # w_tgt, h_tgt = int(downsample_ratio*w), int(downsample_ratio*h)
        # tgt_rgb = tgt_rgb.resize((w_tgt,h_tgt), resample=Image.BICUBIC)

        if not outpainting:
            w_tgt, h_tgt = int(downsample_ratio * w), int(downsample_ratio * h)
            tgt_rgb = tgt_rgb.resize((w_tgt, h_tgt), resample=Image.BICUBIC)
        else:
            w_o, h_o = round(op_f * w), round(op_f * h)
            area = (w_o, h_o, w - w_o, h - h_o)
            src_rgb = src_rgb.crop(area)

            tgt_rgb = tgt_rgb.crop(area)
            w_tgt, h_tgt = int(downsample_ratio * (1 - op_f * 2) * w), int(
                downsample_ratio * (1 - op_f * 2) * h
            )
            tgt_rgb = tgt_rgb.resize((w_tgt, h_tgt), resample=Image.BICUBIC)

        # load depth and convert to disparity PIL image
        src_depth = np.load(src_depth_path)

        src_disp = 1.0 / src_depth
        src_disp[src_depth == -1] = 0

        if self.disp_norm:
            src_disp = (src_disp * 255 / np.max(src_disp)).astype("uint8")
            src_disp = Image.fromarray(src_disp).convert("L")
        else:
            src_disp = Image.fromarray(src_disp).convert("F")
        src_disp = src_disp.resize((w, h), resample=Image.BICUBIC)

        # get camera extrinsic matrix
        rel_pose = self.extrinsics[index][pose_ind]

        data_item = dict()
        data_item["src_image"] = src_rgb
        data_item["tgt_image"] = tgt_rgb
        data_item["src_disp"] = src_disp
        data_item["rel_pose"] = rel_pose
        data_item["src_k"] = src_k
        data_item["tgt_k"] = tgt_k
        data_item["mask"] = src_mask  # np.zeros((h,w), dtype='uint8') # ToDo
        #        data_item["mask"]                = np.zeros((h,w), dtype='uint8') # ToDo
        data_item["src_img_outpainting"] = src_rgb_outpainting

        return data_item

    def _transform(self, data_item):
        return data_item

    def __getitem__(self, idx):
        item_files = self.files[idx]
        data_item = self._read_data(
            self.phase, item_files, idx, self.downsample_ratio
        )

        data_item = self.transformer(data_item)

        splitted_data = (
            data_item["src_image"],
            data_item["tgt_image"],
            data_item["src_disp"],
            data_item["rel_pose"],
            data_item["src_k"],
            data_item["tgt_k"],
            data_item["mask"],
            data_item["src_img_outpainting"],
        )

        return splitted_data


def create_TartanAirLoader(
    tartanairDir,
    phase,
    downsample_ratio=0.5,
    high_gpu=True,
    batch_size=4,
    nthreads=0,
):
    #    if not phase in ['train', 'test', 'val', 'subtrain']:
    #        raise ValueError("Panic::Invalid phase parameter")
    #    else:
    #        pass

    # transf_phase = "train" if phase == "subtrain" else phase
    transf_phase = phase.split("_")[0]

    dataset = TartanAirLoader(tartanairDir, phase, downsample_ratio)

    return DataLoader(
        dataset,
        batch_size,
        shuffle=(transf_phase == "train"),
        num_workers=nthreads,
        pin_memory=high_gpu,
    )
