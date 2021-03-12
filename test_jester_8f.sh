cuda_id=$1

python3 test.py --is_shift --dataset jester --clip_len 8 --shift_div 8 \
                --cuda_id $cuda_id --batch_size 1 --test_crops 3 --scale_size 256 \
                --crop_size 256 --clip_num 10