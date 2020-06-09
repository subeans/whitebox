#!/bin/bash

INSTANCE_TYPE=$1
DATASET=$2
MODEL=$3

# ami = Deep Learning AMI (Ubuntu 18.04) Version 29.0 - ami-0bcc87dc2b59706d6
#key-name = subin_mlpredict.pem
#subnet-id??
#security-group 미리 만들어놓는거 맞나 


LAUNCH_INFO=$(aws ec2 run-instances --image-id ami-0bcc87dc2b59706d6 --count 1 --instance-type $INSTANCE_TYPE \
--key-name subin_mlpredict --subnet-id subnet-abcd1234 --security-group-ids sg-0f4eafe1dc884c02a)

# Instance ID and Public DNS Parsing
sleep 60
INSTANCE_ID=$(echo $LAUNCH_INFO | jq -r '. | .Instances[0].InstanceId')
INSTANCE_PUB_DNS=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID | jq -r '. | .Reservations[0].Instances[0].PublicDnsName')
echo $INSTANCE_PUB_DNS

# Setting for Deep Learning
sleep 60
echo 'setting start'
ssh -i subin_mlpredict.pem -t ubuntu@$INSTANCE_PUB_DNS 'sudo bash rm /usr/local/cuda;ln -s /usr/local/cuda-9.0/ /usr/local/cuda;'

echo 'clone start'
ssh -o "StrictHostKeyChecking no" -i subin_mlpredict.pem ubuntu@$INSTANCE_PUB_DNS 'https://github.com/subeans/ml-performance-prediction.git
'
ssh -i subin_mlpredict.pem -t ubuntu@$INSTANCE_PUB_DNS 'cd ml-performance-prediction/benchmark'
ssh -o "StrictHostKeyChecking no" -i subin_mlpredict.pem ubuntu@$INSTANCE_PUB_DNS 'https://github.com/subeans/whitebox.git'


# Run Experiments
sleep 10
echo 'run start'
BASE_COMMAND="cd ml-performance-prediction/benchmark;docker build -t mlbenchmark . "
RUN_COMMAND="$BASE_COMMAND$INSTANCE_TYPE"
ssh -i subin_mlpredict.pem -t ubuntu@$INSTANCE_PUB_DNS $RUN_COMMAND

# Get csv files from instance
sleep 10
mkdir $INSTANCE_TYPE
scp -i subin_mlpredict.pem \
ubuntu@$INSTANCE_PUB_DNS:~/ml-performance-prediction/benchark/benchmark_* ./$INSTANCE_TYPE/
 
# Terminate Instance
sleep 10
TERMINATE_INFO=$(aws ec2 terminate-instances --instance-ids $INSTANCE_ID)
echo $TERMINATE_INFO
