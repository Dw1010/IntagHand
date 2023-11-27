set -e
export PYTHONPATH="/root/work/IntagHand":${PYTHONPATH}
in_path=/mnt/nas/share-all/caizebin/03.dataset/handpose/images/20231127T163020+0800_ema_001/CAMERA_BOTTOM_RIGHT
out_path=/mnt/nas/share-all/caizebin/03.dataset/handpose/est_images/20231127T163020+0800_ema_001/CAMERA_BOTTOM_RIGHT
mkdir -p ${out_path}
config=/mnt/nas/share-all/caizebin/04.model/handpose/intaghand/misc/model/config.yaml
python apps/demo.py \
    --img_path ${in_path} \
    --save_path ${out_path} \
    --cfg ${config}
