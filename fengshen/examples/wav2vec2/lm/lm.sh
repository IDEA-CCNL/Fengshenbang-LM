#!/bin/bash
#SBATCH --job-name=3-gram # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=4 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

src=/cognitive_comp/common_data/wenetspeech_untar/WenetSpeech.json
tgt_dir=/cognitive_comp/zhuojianheng/data/wenet/lm_data
model_home=/cognitive_comp/zhuojianheng/data/wenet/lm_model
text=$tgt_dir/lm_data.txt

python src/lm_data.py --src $src --tgt_dir $tgt_dir
exec_home=/cognitive_comp/zhuojianheng/work/git/kenlm/build/bin
$exec_home/lmplz -o 3 --verbose header --text $text --arpa $model_home/test.arpa
$exec_home/build_binary trie -a 22 -q 8 -b 8 $model_home/test.arpa $model_home/test.bin

# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ /cognitive_comp/gaoxinyu/docker/flashlight-v1.sif $exec_home/lmplz -o 3 --verbose header --text $text --arpa $model_home/test.arpa
# singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ /cognitive_comp/gaoxinyu/docker/flashlight-v1.sif $exec_home/build_binary trie -a 22 -q 8 -b 8 $model_home/test.arpa $model_home/test.bin
