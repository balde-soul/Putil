## gongsi data
### walking: 0
### run: 1
### jump: 2

python Putil/tools/data_process/yolo/labelme_class_sum.py \
--xml_root /home/Project/0228run/Annotations/
# 生成Keypoint.pkl
touch /home/Project/0228run/KeypointV1-ForSport.pkl \
&& rm /home/Project/0228run/KeypointV1-ForSport.pkl \
&& python mmaction2/tools/data/skeleton/skeleton_extract_cjh.py \
--output /home/Project/0228run/KeypointV1-ForSport.pkl \
--target_class run \
--onehot 1 \
--image_root /home/Project/0228run/JPEGImages/ \
--base_voc_det \
--voc_xml_root /home/Project/0228run/Annotations/
# todo: 分割数据集

python Putil/tools/data_process/yolo/labelme_class_sum.py \
--xml_root /home/Project/0228jump/Annotations/
# 生成Keypoint.pkl
touch /home/Project/0228jump/KeypointV1-ForSport.pkl \
&& rm /home/Project/0228jump/KeypointV1-ForSport.pkl \
&& python mmaction2/tools/data/skeleton/skeleton_extract_cjh.py \
--output /home/Project/0228jump/KeypointV1-ForSport.pkl \
--target_class jump \
--onehot 2 \
--image_root /home/Project/0228jump/JPEGImages/ \
--base_voc_det \
--voc_xml_root /home/Project/0228jump/Annotations/

python Putil/tools/data_process/yolo/labelme_class_sum.py \
--xml_root /home/Project/0302run/Annotations/
# 生成Keypoint.pkl
touch /home/Project/0302run/KeypointV1-ForSport.pkl \
&& rm /home/Project/0302run/KeypointV1-ForSport.pkl \
&& python mmaction2/tools/data/skeleton/skeleton_extract_cjh.py \
--output /home/Project/0302run/KeypointV1-ForSport.pkl \
--target_class run \
--onehot 1 \
--image_root /home/Project/0302run/JPEGImages/ \
--base_voc_det \
--voc_xml_root /home/Project/0302run/Annotations/

python Putil/tools/data_process/yolo/labelme_class_sum.py \
--xml_root /home/Project/0301jump/Annotations/
# 生成Keypoint.pkl
touch /home/Project/0301jump/KeypointV1-ForSport.pkl \
&& rm /home/Project/0301jump/KeypointV1-ForSport.pkl \
&& python mmaction2/tools/data/skeleton/skeleton_extract_cjh.py \
--output /home/Project/0301jump/KeypointV1-ForSport.pkl \
--target_class jump \
--onehot 2 \
--image_root /home/Project/0301jump/JPEGImages/ \
--base_voc_det \
--voc_xml_root /home/Project/0301jump/Annotations/

## 提取human3.6m
#rm /home/Project/Human3.6M/KeypointV1-ForSport.pkl \
#&& python mmaction2/tools/data/skeleton/human36m_pose_extraction_cjh.py \
#--extract \
#--extract_to /home/Project/Human3.6M/KeypointV1-ForSport.pkl \
#--use_frame -1 \
#--target_class Walking,Walk \
#--onehot 0

## 提取ntu
#touch /home/Project/NTU-RGBD/KeypointV1-ForSport.pkl \
#&& rm /home/Project/NTU-RGBD/KeypointV1-ForSport.pkl \
#&& python mmaction2/tools/data/skeleton/ntu_pose_extraction_cjh.py \
#--extract \
#--extract_to /home/Project/NTU-RGBD/KeypointV1-ForSport.pkl \
#--target_class 027 099 026 \
#--onehot 2 1 3


#python Putil/tools/data_process/pkl_combine.py \
#--in_pkls \
#/home/Project/0228run/KeypointV1-ForSport.pkl \
#/home/Project/0228jump/KeypointV1-ForSport.pkl \
#/home/Project/0301jump/KeypointV1-ForSport.pkl \
#/home/Project/0302run/KeypointV1-ForSport.pkl \
#/home/Project/NTU-RGBD/KeypointV1-ForSport.pkl \
#/home/Project/Human3.6M/KeypointV1-ForSport.pkl \
#--out_pkl \
#/home/Project/TsportSkeleton/Alltrain.pkl \
#--pkl_type list

# 针对未标注图像进行数据提取
mkdir -p /home/Project/20230311人员走路样本/JPEGImages-ForSport/ \
&& rm -r /home/Project/20230311人员走路样本/JPEGImages-ForSport/ \
&& VideoDir=/home/Project/20230311人员走路样本/video/ FPS=1.0 IFS="" \
&& ls ${VideoDir} | while read -r line || [ -n "${line}" ]; do TargetVideo=${VideoDir}${line} \
&& python Putil/tools/data_process/extract_frame.py \
--video_path ${TargetVideo} \
--save_to /home/Project/20230311人员走路样本/JPEGImages-ForSport --fps ${FPS}; done
python mmaction2/tools/data/skeleton/skeleton_extract_cjh.py \
--do_visual_sample \
--visual_sample_clip 0 1 2 \
--image_root /home/Project/20230311人员走路样本/JPEGImages-ForSport/ \
--target_class 'person' \
--det_score_conf 0.8 \
--onehot 0 \
--output /home/Project/20230311人员走路样本/KeypointV1-ForSport.pkl

# <block_begin: 20230801增加
# @time 2023-08-01
# @author cjh
touch /home/Project/剧烈运动-20230731103624/KeypointV1-ForSport.pkl \
&& rm /home/Project/剧烈运动-20230731103624/KeypointV1-ForSport.pkl \
&& python mmaction2/tools/data/skeleton/skeleton_extract_cjh.py \
--do_visual_sample \
--visual_sample_clip 0 1 2 \
--image_root /home/Project/剧烈运动-20230731103624/JPEGImages-ForSport/ \
--target_class 'person' \
--det_score_conf 0.8 \
--onehot 0 \
--output /home/Project/剧烈运动-20230731103624/KeypointV1-ForSport.pkl
touch /home/Project/剧烈运动大模型-20230731103648/KeypointV1-ForSport.pkl \
&& rm /home/Project/剧烈运动大模型-20230731103648/KeypointV1-ForSport.pkl \
&& python mmaction2/tools/data/skeleton/skeleton_extract_cjh.py \
--do_visual_sample \
--visual_sample_clip 0 1 2 \
--image_root /home/Project/剧烈运动大模型-20230731103648/JPEGImages-ForSport/ \
--target_class 'person' \
--det_score_conf 0.8 \
--onehot 0 \
--output /home/Project/剧烈运动大模型-20230731103648/KeypointV1-ForSport.pkl
# block_end: >

# 合并
touch /home/Project/TsportSkeleton/train-v3.pkl \
&& rm /home/Project/TsportSkeleton/train-v3.pkl \
&& touch /home/Project/TsportSkeleton/val-v3.pkl \
&& rm /home/Project/TsportSkeleton/val-v3.pkl \
&& python Putil/tools/data_process/pkl_combine.py \
--in_pkls \
/home/Project/0228run/KeypointV1-ForSport.pkl \
/home/Project/0228jump/KeypointV1-ForSport.pkl \
/home/Project/0301jump/KeypointV1-ForSport.pkl \
/home/Project/0302run/KeypointV1-ForSport.pkl \
/home/Project/20230311人员走路样本/KeypointV1-ForSport.pkl \
/home/Project/剧烈运动-20230731103624/KeypointV1-ForSport.pkl \
/home/Project/剧烈运动大模型-20230731103648/KeypointV1-ForSport.pkl \
--out_pkl \
/home/Project/TsportSkeleton/train-v3.pkl \
--remain_pkl \
/home/Project/TsportSkeleton/val-v3.pkl \
--do_balance \
--pkl_type list

:'
nohup python tools/train.py ../config/sport_stgcn_v3.py --gpu-ids 2 --seed 1995 &
'

:'
IFS="" && ImagePath=V0E0Images Tag=ForSportSkeleton-v3E2E300 RootDir=/home/Project/Test/Tsport/ && ls ${RootDir}${ImagePath}/ | while read -r line || [ -n "${line}" ]; do nohup python api/all_type_test.py --source ${RootDir}${ImagePath}/${line} --result_save_to ${RootDir}result_${ImagePath} --tag ${Tag} >> ${RootDir}${Tag}.log & done
IFS="" && VideoPath=20230213剧烈运动误检/video SampleRate=0.3 Tag=ForSportSkeleton-v3E2E300 RootDir=/home/Project/Test/Tsport/ && ls ${RootDir}/${VideoPath} | while read -r line || [ -n "${line}" ]; do nohup python api/all_type_test.py --source ${RootDir}${VideoPath}/${line} --result_save_to ${RootDir}result_${VideoPath} --sample_rate ${SampleRate} --size_sample_rate 0.7 --tag ${Tag}>> ${RootDir}${Tag}.log & done
IFS="" && VideoPath=20230223剧烈运动测试样本/video SampleRate=0.3 Tag=ForSportSkeleton-v3E2E300 RootDir=/home/Project/Test/Tsport/ && ls ${RootDir}/${VideoPath} | while read -r line || [ -n "${line}" ]; do nohup python api/all_type_test.py --source ${RootDir}${VideoPath}/${line} --result_save_to ${RootDir}result_${VideoPath} --sample_rate ${SampleRate} --size_sample_rate 0.7 --tag ${Tag}>> ${RootDir}${Tag}.log & done
IFS="" && VideoPath=AI中台接入01_20230128113527_20230128114326_1/video SampleRate=0.3 Tag=ForSportSkeleton-v3E2E300 RootDir=/home/Project/Test/Tsport/ && ls ${RootDir}/${VideoPath} | while read -r line || [ -n "${line}" ]; do nohup python api/all_type_test.py --source ${RootDir}${VideoPath}/${line} --result_save_to ${RootDir}result_${VideoPath} --sample_rate ${SampleRate} --size_sample_rate 0.7 --tag ${Tag}>> ${RootDir}${Tag}.log & done
IFS="" && VideoPath=20230315剧烈运动测试样本/video SampleRate=0.3 Tag=ForSportSkeleton-v3E2E300 RootDir=/home/Project/Test/Tsport/ && ls ${RootDir}/${VideoPath} | while read -r line || [ -n "${line}" ]; do nohup python api/all_type_test.py --source ${RootDir}${VideoPath}/${line} --result_save_to ${RootDir}result_${VideoPath} --sample_rate ${SampleRate} --size_sample_rate 0.7 --tag ${Tag}>> ${RootDir}${Tag}.log & done
IFS="" && VideoPath=20230315剧烈运动测试样本/video3 SampleRate=0.3 Tag=ForSportSkeleton-v3E2E300 RootDir=/home/Project/Test/Tsport/ && ls ${RootDir}/${VideoPath} | while read -r line || [ -n "${line}" ]; do nohup python api/all_type_test.py --source ${RootDir}${VideoPath}/${line} --result_save_to ${RootDir}result_${VideoPath} --sample_rate ${SampleRate} --size_sample_rate 0.7 --tag ${Tag}>> ${RootDir}${Tag}.log & done
IFS="" && VideoPath=20230315剧烈运动测试样本/video4 SampleRate=0.3 Tag=ForSportSkeleton-v3E2E300 RootDir=/home/Project/Test/Tsport/ && ls ${RootDir}/${VideoPath} | while read -r line || [ -n "${line}" ]; do nohup python api/all_type_test.py --source ${RootDir}${VideoPath}/${line} --result_save_to ${RootDir}result_${VideoPath} --sample_rate ${SampleRate} --size_sample_rate 0.7 --tag ${Tag}>> ${RootDir}${Tag}.log & done
IFS="" && VideoPath=20230315剧烈运动测试样本/video-public SampleRate=1.0 Tag=ForSportSkeleton-v3E2E300 RootDir=/home/Project/Test/Tsport/ && ls ${RootDir}/${VideoPath} | while read -r line || [ -n "${line}" ]; do nohup python api/all_type_test.py --source ${RootDir}${VideoPath}/${line} --result_save_to ${RootDir}result_${VideoPath} --sample_rate ${SampleRate} --size_sample_rate 0.7 --tag ${Tag} --snapet_target run jump >> ${RootDir}${Tag}.log & done
IFS="" && VideoPath=20230411剧烈运动测试/video SampleRate=1.0 Tag=ForSportSkeleton-v3E2E300 RootDir=/home/Project/Test/Tsport/ && ls ${RootDir}/${VideoPath} | while read -r line || [ -n "${line}" ]; do nohup python api/all_type_test.py --source ${RootDir}${VideoPath}/${line} --result_save_to ${RootDir}result_${VideoPath} --sample_rate ${SampleRate} --size_sample_rate 0.7 --tag ${Tag} --snapet_target run jump >> ${RootDir}${Tag}.log & done
VideoPath=演示视频/ Tag=ForAction SampleRate=1.0 RootDir=/home/Project/Test/Tsport/ && ls ${RootDir}${VideoPath}/ | while read -r line || [ -n "${line}" ]; do nohup python api/all_type_test.py --source ${RootDir}${VideoPath}/${line} --result_save_to ${RootDir}result_${VideoPath} --sample_rate ${SampleRate} --size_sample_rate 1.0 --tag ${Tag} --snapet_target no_cap_head >> ${RootDir}${Tag}.log & done
'
:'
# 通过nginx进行均衡nginx监听9000
nohup python api.py --port 9001 &
nohup python api.py --port 9002 &
nohup python api.py --port 9003 &
nohup python api.py --port 9004 &
nohup python api.py --port 9005 &
nohup python api.py --port 9006 &
'

:'
zip ../剧烈运动-api-standardv2-V3E2E300.zip -r * -x '__pycache__/*' -x '*/__pycache/*' -x './mmaction2/work_dirs/*' -x *.gif -x './mmaction2/tools/*' -x './mmaction2/test/*' -x './mmaction2/demo'-x' ./log/*'
'
# scp -o "ProxyCommand ncat --proxy-type socks4 --proxy-auth inspur:Inspur111222 --proxy 10.110.63.27:11080 192.168.12.145 22" -r 20230303白色透明防尘帽/ root@192.168.12.145:/home/Project/

## 测试样本集
### 奔跑
#### 水平视角
#* https://www.youtube.com/watch?v=MoxFkJlVZlA
#* https://www.youtube.com/watch?v=DDM8N88FOo8
#* https://www.youtube.com/watch?v=deX4bDzyOWc
#### 监控视角
#* https://www.youtube.com/watch?v=IhVq9X-Y9hY
#* https://www.youtube.com/watch?v=Kzee0gPqcwE
### 跳跃
### 监控视角
#* https://www.youtube.com/watch?v=xe7uDlJz8HE
#* https://www.youtube.com/watch?v=gg-ciHf1YXU
