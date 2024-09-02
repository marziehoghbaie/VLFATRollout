#!/usr/bin/env bash

%runscript
  exec "$@"

cd /optima/exchange/Marzieh/VLFATRollOut
echo "Where am I ..."
ls
nvidia-smi


python main/Smain.py --config_path /optima/exchange/Marzieh/VLFATRollOut/config/YML_files/FATRollOut.yaml
python main/Smain.py --config_path /optima/exchange/Marzieh/VLFATRollOut/config/YML_files/VLFATRollout.yaml
