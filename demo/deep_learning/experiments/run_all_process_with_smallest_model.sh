# 获取shell参数
usage()
{
cat << EOF
    usage: $0 options

    OPTIONS:
       -g       specify the gpus, seperate by ,（指定gpus，使用逗号分隔）
       -w       specify the number of worker for every dataset,（指定数据进程数）
       --help   Usage
EOF
}
batch_size=64
n_worker=1
device_amount=1
gpus=0 #
while getopts "g:b:w:" OPT; do
    case $OPT in
        g) 
            echo "set gpus: $OPTARG"
            array=(${OPTARG//,/ })
            gpus=${OPTARG//,/ }
            device_amount=${#array[*]}
            ;;
        b)
            echo "set batch size: $OPTARG"
            batch_size=$OPTARG
            ;;
        w)
            echo "set n_worker: $OPTARG"
            n_worker=$OPTARG
            ;;
        ?)
            usage
            exit
            ;;
    esac
done
auto_save_source= \
auto_stop_source= \
lr_reduce_source= \
dataset_source= \
data_loader_source= \
data_sampler_source= \
encode_source= \
backbone_source= \
backend_source= \
decode_source= \
loss_source= \
indicator_source= \
statistic_indicator_source= \
optimization_source= \
aug_sources= \
data_type_adapter_source= \
fit_data_to_input_source= \
fit_decode_to_result_source= \
auto_save_name= \
auto_stop_name= \
lr_reduce_name= \
dataset_name= \
data_loader_name= \
data_sampler_name= \
encode_name= \
backbone_name= \
backend_name= \
decode_name= \
loss_name= \
indicator_name= \
statistic_indicator_name= \
optimization_name= \
aug_names= \
data_type_adapter_name= \
fit_data_to_input_name= \
fit_decode_to_result_name= \
horovodrun -np $device_amount -H 127.0.0.1:$device_amount --start-timeout=100 python model/main.py \
--batch_size=$batch_size \
--n_worker_per_dataset=$n_worker \
--gpus $gpus \