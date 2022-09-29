#!/bin/bash
#SBATCH --job-name=wenetpretreatment # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # number of tasks to run per node
#SBATCH --cpus-per-task=64 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

SRC=/cognitive_comp/common_data/wenetspeech_untar
PATH_TO_WAVES="/cognitive_comp/common_data/wenetspeech_segment" 

python wenet_segment.py --json ${SRC}/WenetSpeech.json --src $SRC --tgt $PATH_TO_WAVES

MANIFEST_PATH="/cognitive_comp/zhuojianheng/data/wenet/"

python build_train_manifest.py --json $PATH_TO_JSON --file_home $PATH_TO_WAVES --valid-percent 0.01 --tgt_home $MANIFEST_PATH
python build_test_manifest.py --json $PATH_TO_JSON --file_home $PATH_TO_WAVES --tgt_home $MANIFEST_PATH
