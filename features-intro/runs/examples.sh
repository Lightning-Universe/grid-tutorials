#!/bin/bash
grid run --name hello hello.py
grid run --name specified-requirements-pip --dependency_file ./pip/requirements.txt hello.py
grid run --name specified-requirements-conda --dependency_file ./conda/environemnt.yml hello.py
grid run --name attaching-datastore --datastore_name cifar5 --datastore_version 1 datastore.py --data_dir /datastores/cifar5/1

