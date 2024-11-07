# DDPM_voxel


python train_3d.py --batch_size 1 --voxel_resolution 16 --ch 64 --ch_mult 1 2 --num_res_blocks 2

python visualize_sampling.py (중간과정확인)

python sampling_3d.py --ckpt_path /results/diffusion-3d-00-00-000000/last.ckpt
