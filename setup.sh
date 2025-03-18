#!/bin/bash
<<com
Supports colab, kaggle, paperspace, lambdalabs and jarvislabs environment setup
Usage:
bash setup.sh <ENVIRON> <download_data_or_not>
Example:
bash setup.sh jarvislabs true
com

ENVIRON=$1
DOWNLOAD_DATA=$2
PROJECT="WSDM-Cup-Multilingual-Chatbot-Arena-Kaggle"

if [ "$1" == "colab" ]
then
    cd /content/$PROJECT
    
elif [ "$1" == "kaggle" ]
then
    cd /kaggle/working/$PROJECT

elif [ "$1" == "jarvislabs" ]    
then    
    cd /home/$PROJECT

elif [ "$1" == "paperspace" ]
then
    cd /notebooks/$PROJECT

elif [ "$1" == "lambdalabs" ]
then
    cd /home/ubuntu/$PROJECT

elif [ "$1" == "modal" ]
then
    cd /root/$PROJECT
    
else
    echo "Unrecognized environment"
fi

# install deps
pip install -r requirements.txt --upgrade
pip install flash-attn --no-build-isolation
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
source .env
export KAGGLE_USERNAME=$KAGGLE_USERNAME
export KAGGLE_KEY=$KAGGLE_KEY

# change the data id as per the experiment
if [ "$DOWNLOAD_DATA" == "true" ]
then
    # mkdir input/
    # cd input/
    # kaggle competitions download -c wsdm-cup-multilingual-chatbot-arena
    # unzip wsdm-cup-multilingual-chatbot-arena.zip
    # rm wsdm-cup-multilingual-chatbot-arena.zip
    huggingface-cli login --token $HF_TOKEN
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --local-dir "./data" --local-dir-use-symlinks False Gladiator/wsdm-cup-datasets
else
    echo "Data download disabled"
fi