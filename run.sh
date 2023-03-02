CUDA_VISIBLE_DEVICES=2 sh bash_files/pretrain/cifar/rdmbyol.sh cifar10 400 poly target -1 &
CUDA_VISIBLE_DEVICES=2 sh bash_files/pretrain/cifar/rdmsimsiam.sh cifar10 400 poly target -1 &