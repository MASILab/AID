import torch
import subjectlist as subl
import os
import torchsrc
from tools_kw.dataset import get_train_test_data
import argparse
import yaml


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# hyper parameters
# epoch_num = 201
# learning_rate = 0.00001
start_epoch = 0

network = 'UNet3D'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str,
                        default='experiments28.yaml')
    args = parser.parse_args()

    SRC_ROOT = os.path.dirname(os.path.realpath(__file__))
    yaml_config = os.path.join(SRC_ROOT, f'yaml/{args.yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    exp_dir = config['exp_dir']
    input_img_dir = config['input_img_dir']
    calcium_mask_dir = config['input_calcium_mask']
    learning_rate = config['learning_rate']
    res = config['res']
    imsize = config['imsize']
    sample_size = config['sample_size']
    sample_duration = config['sample_duration']
    batch_size = config['batch_size']
    fcnum = config['fcnum']
    networkName = config['networkName']
    add_calcium_mask = config['add_calcium_mask']
    data_augmentation = config['data_augmentation']
    dual_network = config['dual_network']
    use_siamese = config['use_siamese']
    ValidateAttention = config['ValidateAttention']
    siamese_coeiff = config['siamese_coeiff']
    clss_num = config['clss_num']

    working_dir = os.path.join(exp_dir, 'working_dir')

    train_dict, test_dict = get_train_test_data(config)

    # load image
    num_workers = 2
    if add_calcium_mask:
        train_set = torchsrc.imgloaders.pytorch_loader_clss3D_calcium(train_dict, num_labels=clss_num,
                                                                      input_root_dir=input_img_dir, calcium_mask_dir=calcium_mask_dir,
                                                                      res=res, imsize=imsize, dual_network=dual_network,
                                                                      data_augmentation=data_augmentation)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_set = torchsrc.imgloaders.pytorch_loader_clss3D_calcium(test_dict, num_labels=clss_num, input_root_dir=input_img_dir, calcium_mask_dir=calcium_mask_dir,
                                                                     res=res,
                                                                     imsize=imsize, data_augmentation=False)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    else:
        train_set = torchsrc.imgloaders.pytorch_loader_clss3D(train_dict, num_labels=clss_num, input_root_dir=input_img_dir,
                                                              res=res, imsize=imsize, dual_network= dual_network, data_augmentation=data_augmentation)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_set = torchsrc.imgloaders.pytorch_loader_clss3D(test_dict, num_labels=clss_num, input_root_dir=input_img_dir, res=res,
                                                             imsize=imsize, data_augmentation=False)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # train_set = torchsrc.imgloaders.pytorch_loader_no255(train_dict,num_labels=lmk_num)
        # train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=1)
        # test_set = torchsrc.imgloaders.pytorch_loader_no255(test_dict,num_labels=lmk_num)
        # test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=1)

        # load network

    resnet_shortcut = 'B'

    if networkName == 'resnet101':
        model = torchsrc.models.resnet101(num_classes=clss_num,
                                          shortcut_type=resnet_shortcut,
                                          sample_size=sample_size,
                                          sample_duration=sample_duration,
                                          fcnum=fcnum)
    elif networkName == 'densenet121':
        model = torchsrc.models.densenet121(sample_size0=sample_size[0],
                                            sample_size1=sample_size[1],
                                            sample_duration=sample_duration,
                                            )
    elif networkName == 'densenet121_twochanel':
        model = torchsrc.models.densenet121_twochanel(
            sample_size0=sample_size,
            sample_size1=sample_size,
            sample_duration=sample_duration,
            # num_classes = clss_num
        )
    elif networkName == 'densenet121_twochanel_cam':
        model = torchsrc.models.densenet121_twochanel_cam(sample_size0=sample_size[0],
                                                          sample_size1=sample_size[1],
                                                          sample_duration=sample_duration,
                                                          # num_classes = clss_num
                                                          )
    elif networkName == 'sononet_grid_attention':
        model = torchsrc.models.sononet_grid_attention(
            nonlocal_mode='concatenation_mean_flow',
            aggregation_mode='concat',
            # num_classes = clss_num
        )
    elif networkName == 'sononet_grid_attention_v2':
        model = torchsrc.models.sononet_grid_attention_v2(
            nonlocal_mode='concatenation_softmax',
            aggregation_mode='concat',
            # num_classes = clss_num
        )
    elif networkName == 'densenetyh':
        model = torchsrc.models.densenetyh(sample_size0=sample_size[0],
                                           sample_size1=sample_size[1],
                                           sample_duration=sample_duration,
                                           # num_classes = clss_num
                                           )
    elif networkName == 'huo_net':
        model = torchsrc.models.huo_net(
            nonlocal_mode='concatenation_softmax',
            aggregation_mode='concat',
            # num_classes = clss_num
        )
    elif networkName == 'huo_net_direct':
        model = torchsrc.models.huo_net_direct(
            nonlocal_mode='concatenation_softmax',
            aggregation_mode='concat',
            # num_classes = clss_num
        )
    elif networkName == 'huo_net_conv1':
        model = torchsrc.models.huo_net_conv1(
            nonlocal_mode='concatenation_softmax',
            aggregation_mode='concat',
            n_classes=clss_num
        )

    out = os.path.join(working_dir, 'ResNet3D_out_0.00001')
    mkdir(out)

    # model = torchsrc.models.VNet()

    print_network(model)
    #
    # load optimizor
    optim = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # load CUDA
    cuda = torch.cuda.is_available()
    #cuda = False
    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed(1)
        model = model.cuda()

    # load trainer
    trainer = torchsrc.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        test_loader=test_loader,
        out=out,
        max_epoch=config['epoch_num'],
        batch_size=batch_size,
        lmk_num=clss_num,
        dual_network=dual_network,
        add_calcium_mask=add_calcium_mask,
        use_siamese=use_siamese,
        siamese_coeiff=siamese_coeiff,
        config=config
    )

    print("==start training==")

    trainer.epoch = config['start_epoch']
    if ValidateAttention:
        trainer.epoch = 84
        trainer.max_epoch = trainer.epoch+1
        trainer.test_epoch()
    else:
        trainer.train_epoch()


if __name__ == '__main__':
    main()
