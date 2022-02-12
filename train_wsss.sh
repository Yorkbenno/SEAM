CUDA_VISIBLE_DEVICES='2' python infer_SEAM.py --weights resnet38_SEAM.pth --out_cam outcam --out_cam_pred outcampred --out_crf outcrf
CUDA_VISIBLE_DEVICES='2' python train_aff.py --weights res38d.pth --la_crf_dir outcrf_4.0 --ha_crf_dir outcrf_24.0 --session_name wsss_aff_nolabel
CUDA_VISIBLE_DEVICES='2' python infer_aff.py --weights wsss_aff_nolabel.pth --cam_dir outcam --out_rw outrw