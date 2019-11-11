#!/bin/bash
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1PzOMpRnBUUqJnC_8pEm8VDFBP3xEmQID" -O "20191106_res34_model_57.pth"
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1lED1VAgjhHjjcoQ-3uEkJ1Kwun53ofQT" -O "20191106_res50_model_112.pth"
wget https://www.dropbox.com/s/q8ssyob24uw8zpe/20191105_wideres50_model_146.tar.xz?dl=1 -O "20191105_wideres50_model_146.tar.xz"
tar -xvf 20191105_wideres50_model_146.tar.xz
python3 model_test.py $1 $2



