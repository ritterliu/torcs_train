#!/bin/bash
if [ ! -d keras_model ] ; then
    echo '>>> Download pre-trained models to train'
    wget 'https://cnbj1.fds.api.xiaomi.com/torcs/models.tar.gz'
    tar -xvf models.tar.gz  -C ../
    rm models.tar.gz
fi

function download_sample() {
    if [ "$(ls camera_data/raw_train/)" ] ; then
        if [ "$(ls camera_data/raw_val/)" ] ; then
            return
        fi
    fi
    echo ">>> Download sample data to raw_train and raw_val"
    cnt=0
    cat camera_data/logs.txt | while read line;
    do
        if [ $cnt -lt 2 ] ; then
            if [ ! -f camera_data/raw_train/$line ] ; then
                echo 'https://cnbj1-fds.api.xiaomi.net/aicontest/camera_data/'$line
                wget -P camera_data/raw_train 'https://cnbj1-fds.api.xiaomi.net/aicontest/camera_data/'$line
            fi
        fi        
        
        if [ $cnt -ge 2 -a $cnt -lt 4 ] ; then
            if [ ! -f camera_data/raw_val/$line ] ; then
                echo 'https://cnbj1-fds.api.xiaomi.net/aicontest/camera_data/'$line
                wget -P camera_data/raw_val 'https://cnbj1-fds.api.xiaomi.net/aicontest/camera_data/'$line
            fi
        fi

        (( cnt++ ))
        if [ $cnt -ge 4 ] ; then
            break
        fi
    done
}

if [ ! -f camera_data/raw_train ] ; then
    mkdir -p camera_data/raw_train
    mkdir -p camera_data/raw_val
    download_sample
fi

function download_all() {
    cat camera_data/logs.txt | while read line;
    do
        if [ ! -f camera_data/raw_train/$line ] ; then
            echo 'https://cnbj1-fds.api.xiaomi.net/aicontest/camera_data/'$line
            wget -P camera_data/raw_train 'https://cnbj1-fds.api.xiaomi.net/aicontest/camera_data/'$line
        fi
    done
}

while true; do
    read -p "Do you wish to download all train data (nearly 14G)? It will take some time. [Y/n]?" yn
    case $yn in
        [Yy]* ) download_all; break;;
        [Nn]* ) echo "Okay, you can download them by running this script again."; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
