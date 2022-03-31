#!bin/bash
grid run --dependency_file ./requirements.txt --name gan-tut --instance_type g4dn.xlarge -- gan.py --gpus 1 --batch-size 256 --max-epochs 6
