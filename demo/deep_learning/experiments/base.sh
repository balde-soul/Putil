######################################################################
# set_env
#   使用全局字典env_dict，读取当前环境变量然后赋予env_dict，不存在的话则
#   维持env_dict的原值，然后使用export设置环境变量
# 使用：
# declare -A env_dict
# source <path of this file>
# set_env
######################################################################
declare -A env_dict
function set_env {
    echo 'function:' $0
    ## 从脚本外获取手动设置的环境变量
    for key in $(echo ${!env_dict[*]}); do
        if [ $(eval echo '$'$key) ]; then
            echo 'manual set' $key 'from' ${env_dict[$key]}'(default)-->' $(eval echo '$'$key)
            env_dict[$key]=$(eval echo '$'$key)
        else
            env_dict[$key]=${env_dict[$key]}
        fi
    done
    ## 设置环境变量
    env_dict_command=
    for key in $(echo ${!env_dict[*]}); do
        env_dict_command=$(echo $env_dict_command $key=${env_dict[$key]})
    done
    echo env_dict_command: $env_dict_command
    export $env_dict_command
    return 0
}