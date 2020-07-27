#!/bin/bash

killall tensorboard
tensorboard --logdir=./runs --bind_all &
python disaster_bert.py
