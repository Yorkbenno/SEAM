CUDA_VISIBLE_DEVICES='0' python train_SEAM.py --weights res38d.pth --session_name twoclass
CUDA_VISIBLE_DEVICES='0' python infer_SEAM.py --weights twoclass.pth --out_cam outcam --out_cam_pred outcampred --out_crf outcrf
CUDA_VISIBLE_DEVICES='0' python train_aff.py --weights res38d.pth --la_crf_dir outcrf_4.0 --ha_crf_dir outcrf_24.0 --session_name twoclass_aff
CUDA_VISIBLE_DEVICES='0' python infer_aff.py --weights twoclass_aff.pth --cam_dir outcam --out_rw outrw
# CUDA_VISIBLE_DEVICES='2' python infer_SEAM.py --weights resnet38_SEAM.pth --out_cam validoutcam --out_cam_pred validoutcampred --out_crf validoutcrf
# python eval_utils.py