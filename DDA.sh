
#### bash for DDA
export PYTHONPATH=$PYTHONPATH:$(pwd)
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
                           --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
                            --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
                            --num 1\
                            --D 4 --N 50 --scale 2\
                            --corruption gaussian_noise --severity 5 \
                            --save_dir /data/majc/TTA_results/DDA/Img-C

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption gaussian_blur --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C

########## 因为Imagenet-C有18个corruption，所以相当于18个实验，
##########如果要用im-r、a、s，只用改 --base_samples 的路径和将 --corruption "" 写成空就能读取

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/imagenet-r \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption '' --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-r

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/imagenet-s \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption '' --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-s

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/imagenet-a \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption '' --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-a

# #### imagenet-C more corruption
# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption brightness --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption contrast --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption defocus_blur --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C


# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption fog --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C
# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption frost --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C


# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption glass_blur --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption motion_blur --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption pixelate --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption saturate --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption shot_noise --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption spatter --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C

# CUDA_VISIBLE_DEVICES=1,2,0 mpiexec -n 3 python DDA/image_adapt/scripts/image_sample.py $MODEL_FLAGS \
#                            --classifier_scale 1.0 --batch_size 4 --num_samples 1000 --timestep_respacing 100 \
#                             --model_path /data/majc/models/DDA/256x256_diffusion_uncond.pt --base_samples /data/majc/ImageNet-C \
#                             --num 1\
#                             --D 4 --N 50 --scale 2\
#                             --corruption zoom_blur --severity 5 \
#                             --save_dir /data/majc/TTA_results/DDA/Img-C
# #