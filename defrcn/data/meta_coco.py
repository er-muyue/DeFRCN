import io
import os
import contextlib
import numpy as np
from pycocotools.coco import COCO
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_meta_coco"]


def load_coco_json(json_file, image_root, metadata, dataset_name):
    is_shots = "shot" in dataset_name  # few-shot
    if is_shots:
        imgid2info = {}
        shot = dataset_name.split('_')[-2].split('shot')[0]
        seed = int(dataset_name.split('_seed')[-1])
        split_dir = os.path.join('datasets', 'cocosplit', 'seed{}'.format(seed))
        for idx, cls in enumerate(metadata["thing_classes"]):
            json_file = os.path.join(split_dir, "full_box_{}shot_{}_trainval.json".format(shot, cls))
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            for img_id in img_ids:
                if img_id not in imgid2info:
                    imgid2info[img_id] = [coco_api.loadImgs([img_id])[0], coco_api.imgToAnns[img_id]]
                else:
                    for item in coco_api.imgToAnns[img_id]:
                        imgid2info[img_id][1].append(item)
        imgs, anns = [], []
        for img_id in imgid2info:
            imgs.append(imgid2info[img_id][0])
            anns.append(imgid2info[img_id][1])
    else:
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    imgs_anns = list(zip(imgs, anns))
    id_map = metadata["thing_dataset_id_to_contiguous_id"]

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"]

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(
            image_root, img_dict["file_name"]
        )
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if obj["category_id"] in id_map:
                obj["category_id"] = id_map[obj["category_id"]]
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_meta_coco(name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_coco_json(annofile, imgdir, metadata, name),
    )

    if "_base" in name or "_novel" in name:
        split = "base" if "_base" in name else "novel"
        metadata["thing_dataset_id_to_contiguous_id"] = metadata[
            "{}_dataset_id_to_contiguous_id".format(split)
        ]
        metadata["thing_classes"] = metadata["{}_classes".format(split)]

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        dirname="datasets/coco",
        **metadata,
    )