# CUDA_VISIBLE_DEVICES='3' python train_SEAM.py --weights res38d.pth
# CUDA_VISIBLE_DEVICES='3' python infer_SEAM.py --weights crag_seam.pth --out_cam outcam --out_cam_pred outcampred --out_crf outcrf
# CUDA_VISIBLE_DEVICES='3' python train_aff.py --weights res38d.pth --la_crf_dir outcrf_4.0 --ha_crf_dir outcrf_24.0
# CUDA_VISIBLE_DEVICES='3' python infer_aff.py --weights crag_aff.pth --cam_dir outcam --out_rw outrw
CUDA_VISIBLE_DEVICES='2' python infer_SEAM.py --weights resnet38_SEAM.pth --out_cam validoutcam --out_cam_pred validoutcampred --out_crf validoutcrf
python eval_utils.py