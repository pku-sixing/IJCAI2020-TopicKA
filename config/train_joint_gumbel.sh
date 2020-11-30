export CUDA_VISIBLE_DEVICES=1
python -u -m KEFU3.run_kefu3 --config=config/JointGumbel.txt  --pretrain_config_path=config/Joint.txt >> logs/JointGumbel.txt
python -u -m KEFU3.run_kefu3 --config=config/JointGumbel.txt --test True -b 1
python -u eval.py --config=config/JointGumbel.txt -b 1 &