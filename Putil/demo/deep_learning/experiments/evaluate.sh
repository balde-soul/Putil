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
gpus=0
n_worker=1
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
source activate 356
python main.py \
--off_train \
--off_test \
--weight= \
--batch_szie=$batch_size \
--gpus $gpus \
--n_worker_pre_dataset=$n_worker \