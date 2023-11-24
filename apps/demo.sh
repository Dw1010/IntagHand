set -e
export PYTHONPATH="/root/work/IntagHand":${PYTHONPATH}
in_path=./demo
out_path=./output/demo/
mkdir -p ${out_path}
config=/mnt/nas/share-all/caizebin/04.model/handpose/intaghand/misc/model/config.yaml
python apps/demo.py \
    --img_path ${in_path} \
    --save_path ${out_path} \
    --cfg ${config}
