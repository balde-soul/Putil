usage()
{
cat << EOF
    usage: [environ data] $0 options

    environ data:
        环境变量，所有同等地位的变量需要以'.'分隔，因为环境变量不接受空格，比如aug_sources与aug_names就存在多种并列的情况，
        需要使用aug_sources=source1.source2.source3与aug_names=name1.name2.name3的形式
        log_level: 
            使用log_level=Info等设定log等级
        del_train_time: 
            使用del_train_time=tim-1.time-2.time-3, 删除训练总目录中的分段训练目录
        train_name: 
            使用train_name=*name*，设置训练代表性目的以及影响训练子目录的名称

        name: 
            the ${backbone_name}${name} would be the name of the fold to save the result
            默认Unnamed
        remote_debug: 
            使用remote_debug=True或者remote_debug=true设定环境变量，代表即将进行remote_debug模式
            setup with remote debug(blocked while not attached) or not'
            默认False
        run_stage: 
            代表着本次运行的RunStage，默认是Train
            Train/train-->train_off=False，evaluate_off=*，test_off=*
            Evaluate/evaluate-->train_off=True，evaluate_off=False
            Test/test-->train_off=True，evaluate_off=True，test_off=False
        save_dir:
            最基础的保存结果的根目录
            默认为./result
        weight_path: 
            可以指定权重文件存放位置,
            加上weight_epoch的指定，可以生成继续训练所继承的weight
            help=the path where the trained model saved
            默认为None，代表None，那么本次运行将不会是继续训练
        weight_epoch: 
            作用查看weight_path
            help=the epoch when saved model which had been trained
            默认为'None'代表None，那么本次运行将不会是继续训练
        train_name:
            继续训练的文件夹主名称
            help=specify the name as the prefix for the continue train fold name
            默认为''
        debug:
            help='run all the process in two epoch with tiny data')
            默认为False
        clean_train:
            使用A.B.C的模式设置的一个列表，表示即将被清理的train_time文件夹
            help=if not None, the result in specified train time would be clean, need args.weight_path
            默认为
        framework:
            指定使用框架：
            Torch/torch-->torch
            tf/tensorflow-->tf
        log_level:
            指定log等级
            Debug、Info、Warning、Error、Fatal
            默认Info

    OPTIONS:
        -g      （指定gpus，使用逗号分隔）ip0:gpu0.gpu1,ip1:gpu0.gpu1 example: 127.0.0.1:0.1.2,127.0.0.2:0,127.0.0.3:0.1
        -w       specify the number of worker for every dataset,（指定数据进程数）
        -b       specify the batch size
        --help   Usage
EOF
}

function print_dict(){
    adict=$1
    for key in $(echo ${!adict[*]}); do 
        echo $key=${adict[$key]}; 
    done
    return 0
}

batch_size=2
n_worker=1
gpus=()
ips=()
base_lr=0.001
while getopts "g:b:w:l:" OPT; do
    case $OPT in
        g) 
            echo 'set gpus: $OPTARG'
            ip_gpu_array=(${OPTARG//,/ })
            for (( i=0;i<${#ip_gpu_array[@]};i++)); do
                echo $i
                #echo ${ip_gpu_array[i]}
                ip_gpu=(${ip_gpu_array[i]//:/ })
                # 获取一个ip加一组gpu，如果没有指定ip，那么默认为127.0.0.1， 如果127.0.0.1已经存在了，那么一场结束
                if [ ${#ip_gpu[@]} -eq 1 ]; then 
                    echo 'only specify the gpu'
                    ip=127.0.0.1
                    gpu=${ip_gpu[0]}
                else
                    ip=${ip_gpu[0]}
                    gpu=${ip_gpu[1]}
                fi
                amount=(${gpu//./ })
                amount=${#amount[@]}
                gpus[$i]=$gpu
                ips[$i]=$ip
                #if [ $i -ne 0] then
                #    if [[ "$list_name" =~ "$var" ]]
                #    then
                #        ${var}
                #    fi
                #fi
                echo 'get ip:gpu:amount' ${ips[$i]}':' ${gpus[$i]}':' $amount
            done
            ;;
        b)
            echo "set batch size: $OPTARG"
            batch_size=$OPTARG
            ;;
        w)
            echo "set n_worker: $OPTARG"
            n_worker=$OPTARG
            ;;
        l)
            echo "set base_lr: $OPTAVG"
            base_lr=$OPTARG
            ;;
        ?)
            usage
            exit
            ;;
    esac
done

source ./experiments/base.sh
# 获取horovod中的H,np与main中的gpus参数
gpus_arg=
horovod_np_arg=0
horovod_H_arg=
if [ ${#gpus[@]} -eq 0 ] && [ ${#ips[@]} -eq 0 ]; then
    gpus[0]=0
    ips[0]=127.0.0.1
fi
for (( i=0;i<${#gpus[@]};i++ )); do
    gpus_arg=$(echo $gpus_arg ${gpus[$i]})
    echo 'gpu data:' ${gpus[$i]}
    amount=(${gpus[$i]//./ })
    amount=${#amount[@]}
    echo amount: $amount
    #if [ -z '$horovod_np_arg' ]; then 
    if [ -z "$horovod_H_arg" ]; then
        echo 'empty'
        horovod_H_arg=$horovod_H_arg
    else 
        horovod_H_arg=$(echo $horovod_H_arg,)
    fi
    horovod_H_arg=$(echo $horovod_H_arg${ips[$i]}:$amount)
    echo horovod_H_arg: $horovod_H_arg
    horovod_np_arg=$[$horovod_np_arg+$amount]
done
echo gpus_arg: $gpus_arg horovod_np_arg: $horovod_np_arg horovod_H_arg: $horovod_H_arg

declare -A env_dict 
env_dict=(
[name]= [remote_debug]=False [run_stage]=Train [save_dir]=./result
[weight_path]=None [weight_epoch]=None [train_name]=
[clean_train]= [debug]=False [framework]=torch [log_level]=Info
)
set_env

env_dict=(
[auto_save_source]=standard [auto_save_name]=DefaultAutoSave
[auto_stop_source]=standard [auto_stop_name]=DefaultAutoStop
[lr_reduce_source]=standard [lr_reduce_name]=DefaultLrReduce
[dataset_source]=standard [dataset_name]=DefaultDataset
[data_loader_source]=standard [data_loader_name]=DefaultDataLoader
[data_sampler_source]=standard [data_sampler_name]=DefaultDataSampler
[encode_source]=standard [encode_name]=DefaultEncode
[backbone_source]=standard [backbone_name]=DefaultBackbone 
[backend_source]=standard [backend_name]=DefaultBackend 
[decode_source]=standard [decode_name]=DefaultDecode
[loss_source]=standard [loss_name]=DefaultLoss
[indicator_source]=standard [indicator_name]=DefaultIndicator
[indicator_statistic_source]=standard [indicator_statistic_name]=DefaultIndicatorStatistic
[optimization_source]=standard [optimization_name]=DefaultOptimization
[aug_source]=standard [aug_name]=DefaultAug
[data_type_adapter_source]=standard [data_type_adapter_name]=DefaultDataTypeAdapter
[fit_data_to_input_source]=standard [fit_data_to_input_name]=DefaultFitDataToInput
[fit_to_loss_input_source]=standard [fit_to_loss_input_name]=DefaultFitToLossInput
[fit_to_indicator_input_source]=standard [fit_to_indicator_input_name]=DefaultFitToIndicatorInput
[fit_to_decode_input_source]=standard [fit_to_decode_input_name]=DefaultFitToDecodeInput
[fit_decode_to_result_source]=standard [fit_decode_to_result_name]=DefaultFitDecodeToResult
[recorder_source]=standard [recorder_name]=DefaultRecorder
[accumulated_opt_source]=standard [accumulated_opt_name]=DefaultAccumulatedOpt
)
set_env
#################################env_set_command 如何使用######################################
#当运行一个.sh文件或者是shell命令，shell会把当前的环境变量都复制过来，也就是子类和父类的关系。通过以下几个场景解释这个概念。
#证明父能影响子
#直接运行命令export K=V，然后echo $K，能看到输出了V
#写一个shell脚本，echo $K，能看到输出了V
#证明子不能影响父
#在一个shell脚本中export K=V，然后echo $K，能看到输出了V。
#基于1，直接运行命令echo $K，发现输出为空
#如果想让shell脚本中执行的环境变量影响到父环境，那么可以用source来执行
#source xxx.sh
#因为source的脚本是在当前环境下执行的，也就是说没有用子shell来执行（默认用sh xxx.sh是新建一个子shell来运行）。这样就可以让脚本中更改的环境变量影响到系统环境变量。但也只是当前ssh连接下的环境变量，其他连接依然不受影响。如果要更改全局的环境变量，那么可以在/etc/profile中添加export xxxx，更改完后source /etc/profile
#如果想删除该变量，可以用unset xxxx
###important: 子shell-shell-terminal是一个圈套环境，而source或者.这个命令解除了嵌套关系，变成了环境等价关系
#################################env_set_command 如何使用######################################