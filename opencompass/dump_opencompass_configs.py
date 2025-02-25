import os
from argparse import ArgumentParser
from typing import List
from itertools import product

from loguru import logger
from mmengine import Config, mkdir_or_exist


def load_and_dumpe_oc_configs(args) -> List[str]:
    num_automatic_task = 0
    save_dir = './opencompass_configs'
    mkdir_or_exist(save_dir)
    
    cfg = Config.fromfile(args.config_template_file)
    paths = []
    for model_cfg, data_cfg in product(cfg['models'], cfg['datasets']):
        save_path = os.path.join(save_dir, 
                                 f"{model_cfg['abbr'].replace('.', '-')}"
                                 f"@{data_cfg['abbr']}.py")
        automatic_task_cfg = Config(dict(models=[model_cfg]))
        automatic_task_cfg.merge_from_dict(dict(datasets=[data_cfg]))
        automatic_task_cfg.dump(save_path)
    
        logger.info(f'|----------> Save opencompass config file to {save_path}')
        paths.append(save_path)
        num_automatic_task += 1
        
    logger.info(f'|----------> Complete saving {num_automatic_task} opencompass config files')
    
    return paths


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-tc', '--config_template_file', 
                        type=str, 
                        help='the path to opencompass '
                             'config template file')
    
    args = parser.parse_args()
    
    load_and_dumpe_oc_configs(args)