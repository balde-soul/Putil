

dataset->data_sampler->data_loader->fit_data_to_input->backbone->backend->decode->fit_decode_to_result->dataset_save_result
                                

dataset--base_on_dataset_name-->dataset_N


data_sampler--base_on_data_sampler_name-->data_sampler_N


data_loader--base_on_data_loader_name-->data_loader_N


fit_data_to_input--base_on_fit_data_to_input_name-->data_index_N


backbone--base_on_backbone_name-->backbone_N


backend--base_on_backend_name-->backend_N


decode--base_on_decode_name-->decode_N


fit_decode_to_result--base_on_fit_decode_to_result_name-->fit_decode_to_result_N


pack backbone->backend->decode in save and deploy


use state_dict of backbone backend decode optimization auto_save auto_stop and lr_reduce to checkpoint