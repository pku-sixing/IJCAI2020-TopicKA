export CUDA_VISIBLE_DEVICES=2
python -u -m KEFU3.run_kefu3 --config=config/JointSample.txt  --pretrain_config_path=config/Joint.txt >> logs/JointSample.txt
python -u -m KEFU3.run_kefu3 --config=config/JointSample.txt --test True -b 1
python -u eval.py --config=config/JointSample.txt -b 1
