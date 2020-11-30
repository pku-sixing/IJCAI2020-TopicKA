export CUDA_VISIBLE_DEVICES=2
python -u -m KEFU3.run_kefu3 --config=config/RoundSample.txt  --pretrain_config_path=config/Round.txt >> logs/RoundSample.txt
python -u -m KEFU3.run_kefu3 --config=config/RoundSample.txt --test True -b 1
python -u eval.py --config=config/RoundSample.txt -b 1 
