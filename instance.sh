#!bin/bash
INSTANCE_TYPE=$1
DATASET=$2
MODEL=$3


LAUNCH_INFO=$(aws ec2 run-instances --image-id ami-abcdefg --count 1 --instance-type $INSTANCE_TYPE --key-name pem_name --subnet-id subnet-abcdef --security-group-ids sg-abcedfg)

# Instance ID and Public DNS Parsing
sleep 60
INSTANCE_ID=$(echo $LAUNCH_INFO | jq -r '. | .Instances[0].InstanceId')
INSTANCE_PUB_DNS=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID | jq -r '. | .Reservations[0].Instances[0].PublicDnsName')
echo $INSTANCE_PUB_DNS

# Setting for Deep Learning
sleep 60
echo 'setting start'
ssh -o "StrictHostKeyChecking no" -i subin.pem -t ubuntu@$INSTANCE_PUB_DNS 'sudo rm /usr/local/cuda;sudo ln -s /usr/local/cuda-9.0/ /usr/local/cuda;'

echo 'clone start'
ssh -i subin.pem ubuntu@$INSTANCE_PUB_DNS 'git clone https://github.com/subeans/ml-performance-prediction.git
'
ssh -i subin.pem -t ubuntu@$INSTANCE_PUB_DNS 'cd ml-performance-prediction/benchmark;git clone https://github.com/subeans/whitebox.git'



# Run Experiments
sleep 10
echo 'run start'
BASE_COMMAND="cd ml-performance-prediction/benchmark;sudo docker build -t mlbenchmark ." 
ssh -i subin.pem -t ubuntu@$INSTANCE_PUB_DNS $BASE_COMMAND

#ssh -i subin_mlpredict.pem -t ubuntu@$INSTANCE_PUB_DNS 'cd ml-performance-prediction/benchmark;mkdir whitebox8860'
BASE_COMMAND="cd ml-performance-prediction/benchmark;sudo rm -rf data;mkdir data"
ssh -i subin.pem -t ubuntu@$INSTANCE_PUB_DNS $BASE_COMMAND


BASE_COMMAND="cd ml-performance-prediction/benchmark;sudo docker run --name=subin -d --runtime=nvidia -it --mount type=bind,source="/home/ubuntu/ml-performance-prediction/benchmark/data",target=/results mlbenchmark;" 
ssh -i subin.pem -t ubuntu@$INSTANCE_PUB_DNS $BASE_COMMAND

#local 로 다운 
flag=true 
while $flag
do 
 if ssh -i subin.pem -t ubuntu@$INSTANCE_PUB_DNS test -e "~/ml-performance-prediction/benchmark/data/finish.txt";then 
	flag=false
  else
  	echo "Wait"
  	sleep 3
  fi
done


sleep 5
scp -i subin.pem ubuntu@$INSTANCE_PUB_DNS:~/ml-performance-prediction/benchmark/data/* ~/mlpredict-results



#Get csv files from instance
sleep 10
#Terminate Instanc
sleep 10
TERMINATE_INFO=$(aws ec2 terminate-instances --instance-ids $INSTANCE_ID)
echo $TERMINATE_INFO
