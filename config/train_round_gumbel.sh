export CUDA_VISIBLE_DEVICES=1
python -u -m KEFU3.run_kefu3 --config=config/RoundGumbel.txt  --pretrain_config_path=config/Round.txt >> logs/RoundGumbel.txt
python -u -m KEFU3.run_kefu3 --config=config/RoundGumbel.txt --test True -b 1
python -u eval.py --config=config/RoundGumbel.txt -b 1 &
