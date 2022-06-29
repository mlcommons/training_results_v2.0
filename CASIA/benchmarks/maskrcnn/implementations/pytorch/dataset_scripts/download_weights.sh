wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
mkdir $DATASET_DIR/coco2017/models
mv R-50.pkl $DATASET_DIR/coco2017/models


python pickle_coco_annotations.py --root /data/coco2017 --ann_file /data/coco2017/annotations/instances_train2017.json --pickle_output_file /data/coco2017/annotations/instances_train2017.json.pickled
