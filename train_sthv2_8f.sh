cuda_id=$1

python3 train.py --is_train --is_shift \
                 --dataset sthv2 --clip_len 8 \
                 --shift_div 8 --wd 5e-4 --dropout 0.5 \
                 --cuda_id $cuda_id --batch_size 64 --lr_steps 30 40 45 \
                 --lr 1e-2 --base_model resnet50 --epochs 50