rnnt_base=/mnt/data4/work/rnnt
datasets=$rnnt_base/datasets
checkpoints=$rnnt_base/checkpoints
results=$(realpath ./rnnt-logs)
tokenized=$rnnt_base/tokenized
sentencepieces=$rnnt_base/sentencepieces
container=nvcr.io/nvdlfwea/mlperfv20/rnnt:20220509.pytorch

source config_DGXA100_1x8x192x1.sh
CONT=$container DATADIR=$datasets LOGDIR=$results METADATA_DIR=$tokenized NEXP=10 \
  SENTENCEPIECES_DIR=$sentencepieces bash ./run_with_docker.sh
