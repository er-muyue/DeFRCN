import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco14', help='', choices=['coco14', 'voc'])
    parser.add_argument('--config_root', type=str, default='', help='the path to config dir')
    parser.add_argument('--shot', type=int, default=1, help='shot to run experiments over')
    parser.add_argument('--seed', type=int, default=0, help='seed to run experiments over')
    parser.add_argument('--setting', type=str, default='fsod', choices=['fsod', 'gfsod'])
    parser.add_argument('--split', type=int, default=1, help='only for voc')
    args = parser.parse_args()
    return args


def load_config_file(yaml_path):
    fpath = os.path.join(yaml_path)
    yaml_info = open(fpath).readlines()
    return yaml_info


def save_config_file(yaml_info, yaml_path):
    wf = open(yaml_path, 'w')
    for line in yaml_info:
        wf.write('{}'.format(line))
    wf.close()


def main():
    args = parse_args()
    suffix = 'novel' if args.setting == 'fsod' else 'all'

    if args.dataset in ['voc']:
        name_template = 'defrcn_{}_r101_novelx_{}shot_seedx.yaml'
        yaml_path = os.path.join(args.config_root, name_template.format(args.setting, args.shot))
        yaml_info = load_config_file(yaml_path)
        for i, lineinfo in enumerate(yaml_info):
            if '  TRAIN: ' in lineinfo:
                _str_ = '  TRAIN: ("voc_2007_trainval_{}{}_{}shot_seed{}", )\n'
                yaml_info[i] = _str_.format(suffix, args.split, args.shot, args.seed)
            if '  TEST: ' in lineinfo:
                _str_ = '  TEST: ("voc_2007_test_{}{}",)\n'
                yaml_info[i] = _str_.format(suffix, args.split)
        yaml_path = yaml_path.replace('novelx', 'novel{}'.format(args.split))
    elif args.dataset in ['coco14']:
        name_template = 'defrcn_{}_r101_novel_{}shot_seedx.yaml'
        yaml_path = os.path.join(args.config_root, name_template.format(args.setting, args.shot))
        yaml_info = load_config_file(yaml_path)
        for i, lineinfo in enumerate(yaml_info):
            if '  TRAIN: ' in lineinfo:
                _str_ = '  TRAIN: ("coco14_trainval_{}_{}shot_seed{}", )\n'
                yaml_info[i] = _str_.format(suffix, args.shot, args.seed)
    else:
        raise NotImplementedError

    yaml_path = yaml_path.replace('seedx', 'seed{}'.format(args.seed))
    save_config_file(yaml_info, yaml_path)


if __name__ == '__main__':
    main()
