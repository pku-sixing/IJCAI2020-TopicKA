# README

Experimental resources for our IJCAI 2020 Paper "[TopicKA: Generating Commonsense Knowledge-Aware Dialogue Responses Towards the Recommended Topic Fact.](https://www.ijcai.org/Proceedings/2020/521)".

## News

- (2020.11.30) We jsut released our code (beta)
## Preparation

Our code is based on the Tensorflow (1.14.0, Python3.6). 

We reuse some codes/scripts from [Tensorflow-NMT](https://github.com/tensorflow/nmt).

### Tested Environment

 [ACL2020-ConKADI](https://github.com/pku-orangecat/ACL2020-ConKADI) and this project share the same environment.

```
     conda create -p ~/envs/conkadi python=3.6
     conda activate ~/envs/conkadi 
     conda install tensorflow-gpu==1.14.0
``` 

### Datasets

Here we provide the processed datasets. If you need the original datasets, please check the cited papers:

- Chinese Weibo ([Baidu Disk](https://pan.baidu.com/s/1xudsxGRt8XauJfKBXO62LA) code: dgm3 / [Google Drive](https://drive.google.com/file/d/186QwI6EEkFY1yZzYmcm4NLVuY0bQBHse/view?usp=sharing)): It is built upon three previous open Chinese Weibo datasets (please see our paper for detail), and we collected commonsense knowledge facts from [ConceptIO](http://www.conceptnet.io/).

It is worth noting that this "Chinese Weibo" is different from the "Chinese Weibo" of our [ACL2020-ConKADI](https://github.com/pku-orangecat/ACL2020-ConKADI).

In addition, to evaluate the model, you need to download a pre-trained Embeddings:
- Chinese： [Tencent AI Lab Embedding Corpus for Chinese Words and Phrases](https://ai.tencent.com/ailab/nlp/en/embedding.html)


    
  
## Training, Inference, and Evaluation
We provide config/scripts files in (/config). 'A > B' means A should be executed and trained before running B. 

Please see [ACL2020-ConKADI](https://github.com/pku-orangecat/ACL2020-ConKADI)  for understanding the following scripts.

- TwoStage:  train_classifier.sh, train_twostage.sh

- MultiTask: train_joint.sh

- Latent Variable: train_joint.sh > train_joint_sample.sh , or train_round.sh > train_round_sample.sh 

- Gumbel Softmax:  train_joint.sh > train_joint_gumbel.sh , or train_round.sh > train_round_gumbel.sh 

### Note

- The released code/environment/config may involve some changes compared with the internal version. Meanwhile, due to the randomness of parameters and environment, the experimental results may fluctuate.

- If you use other evaluation scripts, the results may be different. In our paper, we uniformly evaluate models using the scripts in this project.

- The pre-trained embeddings cannot cover all appeared words, and thus we use random embeddings; therefore, in terms of Embed-AVG/EX, therefore, such results will have minor differences if you repeat multiple evaluations.


# Citation

If you use our code or data, please kindly cite us in your work.

```
@inproceedings{ijcai2020-521,
  title     = {TopicKA: Generating Commonsense Knowledge-Aware Dialogue Responses Towards the Recommended Topic Fact},
  author    = {Wu, Sixing and Li, Ying and Zhang, Dawei and Zhou, Yang and Wu, Zhonghai},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere},	
  pages     = {3766--3772},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/521},
  url       = {https://doi.org/10.24963/ijcai.2020/521},
}

```