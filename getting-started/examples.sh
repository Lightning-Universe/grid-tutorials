#!bin/bash
grid run --dependency_file ./requirements.txt --name cifar-tut --instance_type 2_m60_8gb --datastore_name cifar5 --localdir --datastore_version 1 -- flash-image-classifier.py --data-dir /datastores/cifar5 --gpus 2 --epochs 4 --learning_rate "uniform(1e-5, 1e-1, 5)"
