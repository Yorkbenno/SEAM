CUDA_VISIBLE_DEVICES='1,2,3' python train_SEAM.py --weights res38d.pth
CUDA_VISIBLE_DEVICES='1,2,3' python infer_SEAM.py --weights resnet38_SEAM.pth --out_cam outcam --out_cam_pred outcampred --out_crf outcrf
