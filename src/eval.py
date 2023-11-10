"Evaluation code"

from datasets.nyu import NYUDepthV2

features = [["image_l"], ["depth_l"]]

ds_root = "/hadatasets/nyu/"

eval_loder = NYUDepthV2(
    dataset_root=ds_root,
    split="validation",
    split_json="configs/nyu.json",
    features=features,
    target_size=(256,256)
)
