# ------------------------------------------------ gswgan privacy analysis ------------------------------------------------------
python kd_mosaic.py \
--gpu 0 --batch_size 64 --lr 1e-2 \
--pipeline=multi_teacher \
--teacher resnet8_t \
--student resnet8_t \
--dataset cifar10 --unlabeled cifar10 \
--epochs 500 --fp16 \
--ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/gswgantest_CB100_SEN0.1_W0.3/ \
--alpha 1.0 --seed 1 --clip -noise 10.0 --w_gan 0.3 --w_algn 0.3 --w_baln 0.3 \
--logfile gswgantest_CB100_SEN0.1_W0.3 --use_maxIters \
--from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/



 
# # ------------------------------------------------ max-PATE with align and balance loss and noise ------------------------------------------------------
# python kd_mosaic.py \
# --gpu 1 --batch_size 64 --lr 6e-2 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/PATE_B64LR6e-2_align_balance_maxvote_addnoise/ \
# --T 10000.1 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --use_pate --lap_scale 100.0 --onehot_vote --max_vote --add_noise \
# --logfile PATE_B64LR6e-2_align_balance_maxvote_addnoise --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/


# ------------------------------------------------- max-PATE with align and balance loss but no noise  -------------------------------------------------------------
# python kd_mosaic.py \
# --gpu 0 --batch_size 64 --lr 6e-2 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/PATE_B64LR6e-2_align_balance_maxvote_nonoise/ \
# --T 10000.1 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --use_pate --lap_scale 100.0 --onehot_vote --max_vote \
# --logfile PATE_B64LR6e-2_align_balance_maxvote_nonoise --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/


# # ---------------------------------------------------- PATE with localweight without noise --------------------------------------------------------------
# python kd_mosaic.py \
# --gpu 1 --batch_size 64 --lr 6e-2 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/PATE_B64LR6e-2_no-noise-local-weight_votehalf/ \
# --T 10000.1 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --use_pate --lap_scale 100.0 --onehot_vote --local_weight --add_noise \
# --logfile PATE_B64LR6e-2_no-noise-local-weight_votehalf --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/



# # ---------------------------------------------------- PATE with cls weight with noise ------------------------------------------------------------------
# python kd_mosaic.py \
# --gpu 0 --batch_size 64 --lr 6e-2 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/PATE_B64LR6e-2_add-noise-no-weight_votehalf_clsweight/ \
# --T 10000.1 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --use_pate --lap_scale 100.0 --onehot_vote --local_cls_weight --add_noise \
# --logfile PATE_B64LR6e-2_add-noise-no-weight_votehalf_clsweight --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/

# ---------------------------------------------------- PATE with cls weight without noise ------------------------------------------------------------------
# python kd_mosaic.py \
# --gpu 1 --batch_size 64 --lr 6e-2 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/PATE_B64LR6e-2_no-noise-no-weight_votehalf_clsweight/ \
# --T 10000.1 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --use_pate --lap_scale 100.0 --onehot_vote --local_cls_weight \
# --logfile PATE_B64LR6e-2_no-noise-no-weight_votehalf_clsweight --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/

# # ----------------------------------------------------- PATE without weight, without noise -------------------------------------------------------
# python kd_mosaic.py \
# --gpu 1 --batch_size 64 --lr 6e-2 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/PATE_B64LR6e-2_no-noise-no-weight_votehalf/ \
# --T 10000.1 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --use_pate --lap_scale 100.0 --onehot_vote \
# --logfile PATE_B64LR6e-2_no-noise-no-weight_votehalf --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/
# --resume /data/repo/code/1sl/DFFK/checkpoints/resnet8t/PATE_B64LR6e-2_with-noise-no-weight/latest.pth


# # ----------------------------------------------------- PATE without weight, no-noise --------------------------------------------------------
# python kd_mosaic.py \
# --gpu 1 --batch_size 64 --lr 6e-2 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/PATE_B64LR6e-2_no-noise-weight/ \
# --T 10000.1 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --use_pate --lap_scale 1000.0 --onehot_vote \
# --logfile PATE_B64LR6e-2_no-noise-weight --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/ \
# --resume /data/repo/code/1sl/DFFK/checkpoints/resnet8t/PATE_B64LR6e-2_no-noise-weight/latest.pth


# ----------------------------------------------------- GS-WGAN gradient-clip + noise --------------------------------------------------------
# python kd_mosaic.py \
# --gpu 1 --batch_size 64 --lr 6e-2 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/clip_TinfB64LR6e-2_E500/ \
# --T 10000.1 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --clip \
# --logfile clip_TinfB64LR6e-2_E500 --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/
# --resume /data/repo/code/1sl/DFFK/checkpoints/resnet8t/clip_TinfB64LR6e-2_E500/latest.pth


# # ------------------------------------ with pate method and dp (onehot label change without localweight ) ---------------------------------------------------------------
# python kd_mosaic.py \
# --gpu 1 --batch_size 64 --lr 6e-2 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/B64E500_onehot_pate_LR6e-2/ \
# --T 0.1 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --use_pate --lap_scale 100.0 --onehot_vote \
# --logfile B64E500_onehot_pate_LR6e-2 --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/

# ------------------------------------ with pate method and dp (onehot label) ---------------------------------------------------------------
# python kd_mosaic.py \
# --gpu 1 --batch_size 64 --lr 6e-2 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/B64E500_onehot_pate_newvote_lw/ \
# --T 0.1 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --use_pate --lap_scale 100.0 --onehot_vote --local_weight \
# --logfile B64E500_onehot_pate_newvote_lw --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/

## ------------------------------------- with pate method and dp (continuous label) -------------------------------------------------------
# python kd_mosaic.py \
# --gpu 0 --batch_size 64 --lr 6e-3 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/PATE100_B64_LRsched0.006_dspd_0.6wgan_Tinf/ \
# --T 10001.0 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --use_pate --lap_scale 100.0 \
# --logfile PATE100_B64_LRsched0.006_dspd_0.6wgan_Tinf --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/


## ----------------------------- without pate methods (original method 82.32) ------------------------------------------------------------
# python kd_mosaic.py \
# --gpu 1 --batch_size 64 --lr 6e-3 \
# --pipeline=multi_teacher \
# --teacher resnet8_t \
# --student resnet8_t \
# --dataset cifar10 --unlabeled cifar10 \
# --epochs 500 --fp16 \
# --ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8t/B64_LRsched0.006_dspd_0.6wgan/ \
# --T 10001.0 --alpha 1.0 --seed 1 --w_gan 0.6 \
# --logfile B64_LRsched0.006_dspd_0.6wgan --use_maxIters \
# --from_teacher_ckpt /home/lsl/Research/DFFK2-main/mosaic_core/checkpoints/pretrained_teachers/a1.0+sd1+e500+b16/