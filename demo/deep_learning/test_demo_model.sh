# 需要进行运行的点：
# first command: train->save,checkpoint,deploy->lr_reduce->not stop->evaluate->test->
# train->no lr_reduce->not stop->evaluate->save,checkpoint,deploy->lr_reduce->stop->
# load checkpoint->first command content->load saved->run evaluate->load save->test
# 获取shell参数
# sources与name的默认环境值
backbone_name=TestDemoBackbone \
backend_name=TestDemoBackend \
loss_name=TestDemoLoss \
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
