CUDA_VISIBLE_DEVICES='2' python infer_SEAM.py --weights resnet38_SEAM.pth --out_cam_pred outcampred --path /home/yyubm/OEEM/classification/glas/1.training/img
python temp.py
# CUDA_VISIBLE_DEVICES='2' python train_SEAM.py --weights res38d.pth --session_name glas_seam