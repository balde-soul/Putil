function extract_env_param {
    ## 从脚本外获取手动设置的环境变量
    for key in $(echo ${!env_params[*]}); do
        if [ $(eval echo '$'$key) ]; then
            echo 'manual set' $key 'from' ${env_params[$key]}'(default)-->' $(eval echo '$'$key)
            env_params[$key]=$(eval echo '$'$key)
        else
            env_params[$key]=${env_params[$key]}
        fi
    done
    return 0
}