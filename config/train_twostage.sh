export CUDA_VISIBLE_DEVICES=1
python -u -m KEFU3.run_kefu3 --config=config/TwoStage.txt >> logs/TwoStage.txt
python -u -m KEFU3.run_kefu3 --config=config/TwoStage.txt >> logs/TwoStage.txt  --test True -b 1 --test_cueword model/Classifier/decoded/test.predicted_golden_fact_position_top1.txt
python -u eval.py --config=config/TwoStage.txt -b 1 &


python -u eval.py --config=config/TwoStage.txt -b 1 &
python -u eval.py --config=config/Round.txt -b 1 &
python -u eval.py --config=config/JointSample.txt -b 1 &
python -u eval.py --config=config/JointGumbel.txt -b 1 &
python -u eval.py --config=config/Joint.txt -b 1 &