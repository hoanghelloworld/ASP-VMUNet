import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1, 2, 3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" # "0, 1, 2, 3"

import timm
from datasets.dataset import NPY_datasets, Polyp_datasets, Isic_datasets
from tensorboardX import SummaryWriter
# from models.vmunet_v2.vmunet_v2 import VMUNet
from models.vmunet.vmunet_v2 import VMUNetV2
from models.UltraLight.UltraLight_VM_UNet import UltraLight_VM_UNet, UltraLight_VM_UNet_pure_mamba, \
    UltraLight_VM_UNet_pure_mamba_huge
from models.UltraLight.UltraLight_Shifted import UltraLight_shifted_huge

from engine import *

import sys
import torch
from torch.utils.data import DataLoader
from utils import *

from configs.config_setting_v2 import setting_config
from configs.config_setting_UltraLight import setting_config as setting_config_UltraLight
from configs.config_setting_UltraLight_pure_mamba import setting_config as setting_config_UltraLight_pure_mamba, \
    setting_config_huge as setting_config_UltraLight_pure_mamba_huge, setting_config_shifted_huge

import warnings

warnings.filterwarnings("ignore")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def test(config, best_weight):

    print("load model")
    model_cfg = config.model_config
    if config.network == 'vmunet_v2-v2':
        print("this is VMUNet_V2")
        model = VMUNetV2(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
            deep_supervision=model_cfg['deep_supervision'],
        )
        model.load_from()
    elif config.network == 'ultraLight':
        print("this is for UltraLight_VM_UNet")
        model = UltraLight_VM_UNet(num_classes=model_cfg['num_classes'],
                                   input_channels=model_cfg['input_channels'],
                                   c_list=model_cfg['c_list'],
                                   split_att=model_cfg['split_att'],
                                   bridge=model_cfg['bridge'], )
    elif config.network == 'ultraLight_pure_mamba':
        print("this is for UltraLight_VM_UNet pure mamba")
        model = UltraLight_VM_UNet_pure_mamba(num_classes=model_cfg['num_classes'],
                                              input_channels=model_cfg['input_channels'],
                                              c_list=model_cfg['c_list'],
                                              split_att=model_cfg['split_att'],
                                              bridge=model_cfg['bridge'], )
    elif config.network == 'ultraLight_pure_mamba_huge':
        print("this is for UltraLight_VM_UNet pure mamba_ huge version")
        model = UltraLight_VM_UNet_pure_mamba_huge(num_classes=model_cfg['num_classes'],
                                                  input_channels=model_cfg['input_channels'],
                                                  c_list=model_cfg['c_list'],
                                                  split_att=model_cfg['split_att'],
                                                  bridge=model_cfg['bridge'], )
    elif config.network == 'ultraLight_shifted_huge':
        print("this is for UltraLight shift huge version")
        model = UltraLight_shifted_huge(num_classes=model_cfg['num_classes'],
                                                  input_channels=model_cfg['input_channels'],
                                                  c_list=model_cfg['c_list'],
                                                  split_att=model_cfg['split_att'],
                                                  bridge=model_cfg['bridge'], )

    else:
        raise Exception('network in not right!')

    print('#----------Testing----------#')
    # best_weight = torch.load(config.work_dir + 'checkpoints/' + best_weight, map_location=torch.device('cpu'))
    best_weight = torch.load(best_weight, map_location=torch.device('cpu'))
    model.load_state_dict(best_weight)

    val_dataset = Isic_datasets(config.data_path, config, train=False, test_dataset=dataset)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.test_batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True
                            )

    # for name in ['isic17','isic18']:
    for name in ['isic17']:
        # val_loader_t = val_loader_dict[name]
        loss = test_one_epoch(
            # val_loader_t,
            val_loader,
            model,
            criterion,
            logger,
            config,
            test_data_name=name
        )
    # os.rename(
    #     os.path.join(checkpoint_dir, 'best.pth'),
    #     os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
    # )


    print(config.work_dir)


if __name__ == '__main__':
    # config = setting_config             # VM-UNet-v2  over
    # config = setting_config_UltraLight            # UltraLight over
    # config = setting_config_UltraLight_pure_mamba   # UltraLight pure Mamba going-300
    config = setting_config_UltraLight_pure_mamba_huge  # UltraLighe pure Mamba huge version
    # config = setting_config_shifted_huge    # UltraLight shifted channel huge version time times

    best_weight = '/home/cheng/muyi/VM-UNetV2-main/results/ultraLight_pure_mamba_huge_isic17_Sunday_14_July_2024_09h_39m_37s/checkpoints/best-epoch215-loss0.2947.pth'

    test(config, best_weight)