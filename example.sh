#!/bin/bash

job_cmd='python -u -m src.main --epochs 200 --Val 20 --cuda --dataSet NormLJ --learn_method unsup'

eval $job_cmd
