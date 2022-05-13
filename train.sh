# CUDA_VISIBLE_DEVICES='3' python train_SEAM.py --weights res38d.pth --session_name crag_seam_112
CUDA_VISIBLE_DEVICES='2' python infer_SEAM.py --weights crag_seam_112.pth --out_cam_pred validoutcampred --path /home/yyubm/WSSS4LUAD/test_CRAG_labelconsider_112
# CUDA_VISIBLE_DEVICES='3' python infer_SEAM.py --weights crag_seam_112.pth --out_cam validoutcam --out_cam_pred validoutcampred_label --path /home/yyubm/WSSS4LUAD/test_CRAG_labelconsider_112
# CUDA_VISIBLE_DEVICES='3' python infer_SEAM.py --weights crag_seam_112.pth --out_cam_pred outcampred --path /home/yyubm/WSSS4LUAD/Dataset_crag/1.training/img
# CUDA_VISIBLE_DEVICES='2' python train_aff.py --weights res38d.pth --la_crf_dir outcrf_4.0 --ha_crf_dir outcrf_24.0 --session_name twoclass_aff
# CUDA_VISIBLE_DEVICES='2' python infer_aff.py --weights twoclass_aff.pth --cam_dir outcam --out_rw outrw
# CUDA_VISIBLE_DEVICES='3' python infer_SEAM.py --weights crag_seam.pth --out_cam outcam --out_cam_pred outcampred  # --out_crf validoutcrf
python eval_utils.py
# CUDA_VISIBLE_DEVICES='0' python infer_SEAM.py --weights glas_seam.pth --out_cam_pred glasoutcampred --path /home/yyubm/WSSS4LUAD/test_glas_labelconsider_112
# python eval_utils.py