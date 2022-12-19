#!/bin/bash
for i in {1..5}; do
	python3 nvidia-stylegan2-ada/projector.py --outdir=outputs --target=data/processed/align-ja_crop.jpg --network=models/ffhq.pkl --seed=$i --save-video False
done;
