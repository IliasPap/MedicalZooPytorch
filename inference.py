import argparse
import os

import torch
import torch.nn.functional as F

# Lib files
import lib.utils as utils
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
from lib.visual3D_temp import non_overlap_padding_v1,test_padding,non_overlap_padding2
from lib.losses3D import DiceLoss
#

def main():
    args = get_arguments()
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)


    # training_generator, val_generator,test_generator, full_volume, test_volume,affine = medical_loaders.generate_datasets(args,
    #                                                                                            path='./datasets')

    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
                                                                                               path='./datasets')

    model, optimizer = medzoo.create_model(args)
    print(model.state_dict().keys())

    criterion = DiceLoss(classes=args.classes)
   # print(full_volume.shape,test_volume.shape)
    # ## TODO LOAD PRETRAINED MODEL
   # print(affine.shape)
    model.restore_checkpoint(args.pretrained)
    if args.cuda:
        model = model.cuda()
        full_volume = full_volume.cuda()
        #test_volume = test_volume.cuda()
        print("Model transferred in GPU.....")

    print(full_volume.shape)
    #output = non_overlap_padding_v1(args,full_volume,model,criterion)
    output = non_overlap_padding2(args,full_volume,model,criterion)
    ## TODO TARGET FOR LOSS

    #print(loss_dice)
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="iseg2019")
    parser.add_argument('--dim', nargs="+", type=int, default=(16,16,16))
    parser.add_argument('--nEpochs', type=int, default=250)

    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--samples_train', type=int, default=1)
    parser.add_argument('--samples_val', type=int, default=1)
    parser.add_argument('--split', type=float, default=0.9)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=2)
    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--terminal_show_freq', default=50000)
    parser.add_argument('--threshold', default=0.0000000000001, type=float)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--normalization', default='global_mean', type=str,
                        help='Tensor normalization: options max, mean, global')
    parser.add_argument('--loadData', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--pretrained',
                        default='./saved_models/UNET3D_checkpoints/iseg2019/13_06___09_40_/13_06___09_40__last_epoch.pth',
                        type=str, metavar='PATH',
                        help='path to pretrained model')

    args = parser.parse_args()

    args.save = '../inference_checkpoints/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    args.tb_log_dir = '../runs/'
    return args


if __name__ == '__main__':
    main()











'''

def overlap_3d_image():
    B, C, D, H, W = 2, 1, 144, 192, 256
    #B, C, D, H, W = 1, 1, 4, 4, 4
    x = torch.randn(B, C, D, H, W)
    print('IMAGE shape ', x.shape)  # [B, C, D, num_of_patches_H,num_of_patches_W, kernel_size,kernel_size]
    kernel_size = 32
    stride = 16
    patches = x.unfold(4, kernel_size, stride)
    print('patches shape ', patches.shape)  # [B, C, D, H, num_of_patches_W, kernel_size]
    patches = patches.unfold(3, kernel_size, stride)
    print('patches shape ', patches.shape)  # [B, C, D, num_of_patches_H,num_of_patches_W, kernel_size,kernel_size]
    patches = patches.unfold(2, kernel_size, stride)
    print('patches shape ', patches.shape)  # [B, C, num_of_patches_D, num_of_patches_H,num_of_patches_W, kernel_size ,kernel_size,kernel_size]
    # patches = patches.unfold()
    # perform the operations on each patchff
    # ...
    B, C, num_of_patches_D, num_of_patches_H,num_of_patches_W, kernel_size ,kernel_size,kernel_size = patches.shape
    # # reshape output to match F.fold input
    patches = patches.contiguous().view(B, C,num_of_patches_D* kernel_size, -1, kernel_size * kernel_size)
    print(patches.shape)
    patches = patches.contiguous().view(B, C,num_of_patches_D* kernel_size, -1, kernel_size * kernel_size)
    print(patches.shape)
    print('slice shape ',patches[:,:,0,:,:].shape)
    slices = []
    for i in range(num_of_patches_D * kernel_size):

        output = F.fold(
              patches[:,:,i,:,:].contiguous().view(B, C * kernel_size * kernel_size,-1), output_size=(H, W), kernel_size=kernel_size, stride=stride)
        #print(output.shape)  # [B, C, H, W]
        slices.append(output)
    image = torch.stack(slices)
    print(image.shape)
    print(image.is_contiguous())
    image = image.permute(1,2,0,3,4).contiguous().view(B,C,-1,H*W)
    print(image.shape)
    output = F.fold(
        image.contiguous().view(B*H*W, C*kernel_size, -1), output_size=(D, 1), kernel_size=kernel_size, stride=stride)
    print(output.shape)  # [B, C, H, W]


'''