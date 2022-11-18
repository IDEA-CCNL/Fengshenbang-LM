
CLUEDATA_PATH=./CLUE_DATA   #CLUE 原始数据路径
UNIDATA_PATH=./data    #处理数据输出路

SCRIPT_PATH=./data_preprocessing

python $SCRIPT_PATH/afqmc_preprocessing.py --data_path=$CLUEDATA_PATH/afqmc_public --save_path=$UNIDATA_PATH/afqmc
python $SCRIPT_PATH/c3_preprocessing.py --data_path=$CLUEDATA_PATH/c3_public --save_path=$UNIDATA_PATH/c3
python $SCRIPT_PATH/chid_preprocessing.py --data_path=$CLUEDATA_PATH/chid_public --save_path=$UNIDATA_PATH/chid
python $SCRIPT_PATH/csl_preprocessing.py --data_path=$CLUEDATA_PATH/csl_public --save_path=$UNIDATA_PATH/csl
python $SCRIPT_PATH/iflytek_preprocessing.py --data_path=$CLUEDATA_PATH/iflytek_public --save_path=$UNIDATA_PATH/iflytek
python $SCRIPT_PATH/ocnli_preprocessing.py --data_path=$CLUEDATA_PATH/ocnli_public --save_path=$UNIDATA_PATH/ocnli
python $SCRIPT_PATH/tnews_preprocessing.py --data_path=$CLUEDATA_PATH/tnews_public --save_path=$UNIDATA_PATH/tnews
python $SCRIPT_PATH/wsc_preprocessing.py --data_path=$CLUEDATA_PATH/cluewsc2020_public --save_path=$UNIDATA_PATH/wsc
python $SCRIPT_PATH/cmrc2018_preprocessing.py --data_path=$CLUEDATA_PATH/cmrc2018_public --save_path=$UNIDATA_PATH/cmrc2018
