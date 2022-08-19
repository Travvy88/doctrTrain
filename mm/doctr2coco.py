import argparse
import mmcv
import numpy as np


def doctr2coco(path):
    data_infos = mmcv.load(path + '/labels.json')

    annotations = []
    images = []
    obj_count = 0
    for idx, filename in enumerate(mmcv.track_iter_progress(data_infos.keys())):
        image = data_infos[filename]

        height, width = mmcv.imread(path + '/images/' + filename).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        for anno in image['polygons']:
            x_min, y_min, x_max, y_max = anno[0][0], anno[0][1], anno[1][0], anno[1][1],

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=None,
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 0, 'name': 't'}])
    mmcv.dump(coco_format_json, path + '/coco.json')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='doctr format to coco')
    parser.add_argument('p', type=str, help='path to text_detection/train(or val) folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    doctr2coco(args.p)