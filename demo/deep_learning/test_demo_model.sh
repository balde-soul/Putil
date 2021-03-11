# 需要进行运行的点：
# first command: train->save,checkpoint,deploy->lr_reduce->not stop->evaluate->test->
# train->no lr_reduce->not stop->evaluate->save,checkpoint,deploy->lr_reduce->stop->
# load checkpoint->first command content->load saved->run evaluate->load save->test
# 获取shell参数
# sources与name的默认环境值
export debug=True
source ./experiments/base.sh
declare -A env_dict
env_dict=(
[K_backbone_name]=DefaultBackbone
[K_backend_name]=DefaultBackend
[k_optimization_name]=DefaultOptimization
[K_auto_save_name]=DefaultAutoSave
[K_auto_stop_name]=DefaultAutoStop
[K_lr_reduce_name]=DefaultLrReduce
[K_aug_name]=DefaultAug 
[K_loss_name]=DefaultLoss
[K_dataset_name]=DefaultDataset
[K_encode_name]=DefaultEncode
[K_data_type_adapter_name]=DefaultDataTypeAdapter
[K_data_sampler_name]=DefaultDataSampler
[K_data_loader_name]=DefaultDataLoader
[K_fit_to_loss_input_name]=DefaultFitToLossInput
[K_fit_to_indicator_input_name]=DefaultFitToIndicatorInput
[K_indicator_name]=DefaultIndicator
[K_indicator_statistic_name]=DefaultIndicatorStatistic
[K_fit_to_decode_input_name]=DefaultFitToDecodeInput
[K_decode_name]=DefaultDecode
)
set_env
source ./experiments/common.sh
horovodrun -np $horovod_np_arg -H $horovod_H_arg --start-timeout=100 python main.py \
--gpus $gpus_arg \
#--batch_size=$batch_size \
#--n_worker_per_dataset=$n_worker \
