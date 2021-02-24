# 需要进行运行的点：
# first command: train->save,checkpoint,deploy->lr_reduce->not stop->evaluate->test->
# train->no lr_reduce->not stop->evaluate->save,checkpoint,deploy->lr_reduce->stop->
# load checkpoint->first command content->load saved->run evaluate->load save->test
# 获取shell参数
# sources与name的默认环境值
export K_backbone_name=DefaultBackbone
export K_backend_name=DefaultBackend
export k_optimization_name=DefaultOptimization
export K_auto_save_name=DefaultAutoSave
export K_auto_stop_name=DefaultAutoStop
export K_lr_reduce_name=DefaultLrReduce
export K_aug_name=DefaultAug 
export K_loss_name=DefaultLoss
export K_dataset_name=DefaultDataset
export K_encode_name=DefaultEncode
export K_data_type_adapter_name=DefaultDataTypeAdapter
export K_data_sampler_name=DefaultDataSampler
export K_data_loader_name=DefaultDataLoader
export K_fit_to_loss_input_name=DefaultFitToLossInput
export K_fit_to_indicator_input_name=DefaultFitToIndicatorInput
export K_indicator_name=DefaultIndicator
export K_indicator_statistic_name=DefaultIndicatorStatistic
export K_fit_to_decode_input_name=DefaultFitToDecodeInput
export K_decode_name=DefaultDecode
source ./experiments/common.sh
horovodrun -np $horovod_np_arg -H $horovod_H_arg --start-timeout=100 python main.py \
--gpus $gpus_arg \
$remote_debug_arg \
$log_level_arg \
$clean_train_time_arg \
$train_name_arg \
--debug \
#--batch_size=$batch_size \
#--n_worker_per_dataset=$n_worker \
