# CUDA_VISIBLE_DEVICES='2' python infer_SEAM.py --weights glas_seam.pth --out_cam_pred glasoutcampred --path /home/yyubm/WSSS4LUAD/test_glas_labelconsider_112
# python eval_utils.py
CUDA_VISIBLE_DEVICES='2' python train_SEAM.py --weights res38d.pth --session_name glas_seam