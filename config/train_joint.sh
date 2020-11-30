export CUDA_VISIBLE_DEVICES=3

python -u -m KEFU3.run_kefu3 --config=config/Joint.txt >> logs/Joint.txt
python -u -m KEFU3.run_kefu3 --config=config/Joint.txt --test True -b 1
python -u eval.py --config=config/Joint.txt -b 1 &
