# Setting up SLURM cluster

## AWS key_pair
Make sure you have a AWS key_pair or create one as follows:

```
export AWS_KEY_NAME=T5_trainer

aws ec2 create-key-pair \
--key-name $AWS_KEY_NAME \
--query KeyMaterial \
--output text > ~/.ssh/$AWS_KEY_NAME.pem

```

## S3 bucket

Create or assign an S3 bucket, this will be required to upload the training data

To create one 
```
export S3_BUCKET_NAME=fsdp-training 
aws s3 mb s3://${S3_BUCKET_NAME}

```
## Upload training data to S3 bucket

```
aws s3 cp "PATH_to_Dataset_files" S3://S3_BUCKET_NAME

```

## Install Pcluster

```
python3 -m pip install --upgrade "aws-parallelcluster"

pcluster version
{
  "version": "3.1.2"
}

Install Node Version Manager and Node.js. It's required due to AWS Cloud Development Kit (CDK) usage for template generation.

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
chmod ug+x ~/.nvm/nvm.sh
source ~/.nvm/nvm.sh
nvm install --lts
node --version

```

## Create Slurm cluster
```
export CLUSTER_NAME=train-cluster

pcluster create-cluster \
--cluster-name $CLUSTER_NAME \
--cluster-configuration cluster_config.yaml

```
or alternatively is you run the command blow, configure wizard prompts you for all of the information required to create a cluster.
```
 pcluster configure --config cluster-config.yaml
```

## Shared file system 
set up a shared file system as SLURM cluster uses it to make sure all files are synchronized between compute nodes.

### copy data from S3 to shared file system

## run the job
sbatch ./slurm/slurm_sbatch_run.sh

