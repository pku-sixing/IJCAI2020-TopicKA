CUDA_VISIBLE_DEVICES=2
nohup python -u -m KEFU3.run_classifier --config=config/Classifier.txt >> logs/Classifier.txt
python -u -m KEFU3.run_classifier --config=config/Classifier.txt --test True -b 1