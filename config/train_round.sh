export CUDA_VISIBLE_DEVICES=1
python -u -m KEFU3.run_kefu3 --config=config/Round.txt >> logs/Round.txt
python -u -m KEFU3.run_kefu3 --config=config/Round.txt --test True -b 1
python -u eval.py --config=config/Round.txt -b 1 &
