
PRED_DATA_PATH=./predict
SUBMIT_DATA_PATH=./submit

SCRIPT_PATH=./predict2submit

python $SCRIPT_PATH/afqmc_submit.py --data_path=$PRED_DATA_PATH/afqmc-M4_predict.json --save_path=$SUBMIT_DATA_PATH/afqmc_predict.json
python $SCRIPT_PATH/c3_submit.py --data_path=$PRED_DATA_PATH/c3_predict.json --save_path=$SUBMIT_DATA_PATH/c311_predict.json
python $SCRIPT_PATH/chid_submit.py --data_path=$PRED_DATA_PATH/chid_predict.json --save_path=$SUBMIT_DATA_PATH/chid11_predict.json
python $SCRIPT_PATH/csl_submit.py --data_path=$PRED_DATA_PATH/csl_predict.json --save_path=$SUBMIT_DATA_PATH/csl_predict.json
python $SCRIPT_PATH/iflytek_submit.py --data_path=$PRED_DATA_PATH/iflytek_predict.json --save_path=$SUBMIT_DATA_PATH/iflytek_predict.json
python $SCRIPT_PATH/ocnli_submit.py --data_path=$PRED_DATA_PATH/ocnli_predict.json --save_path=$SUBMIT_DATA_PATH/ocnli_50k_predict.json
python $SCRIPT_PATH/tnews_submit.py --data_path=$PRED_DATA_PATH/tnews_predict.json --save_path=$SUBMIT_DATA_PATH/tnews11_predict.json
python $SCRIPT_PATH/wsc_submit.py --data_path=$PRED_DATA_PATH/wsc_predict.json --save_path=$SUBMIT_DATA_PATH/cluewsc11_predict.json

python $SCRIPT_PATH/cmrc2018_submit.py --data_path=$PRED_DATA_PATH/cmrc2018_predict.json --save_path=$SUBMIT_DATA_PATH/cmrc2018_predict.json
