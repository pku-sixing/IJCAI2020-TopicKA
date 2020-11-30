from KEFU3.kefu_classifier import Model
from KEFU3.kefu_classifier_rank import Model as RModel
from lib import model_helper
from lib import config_parser
from lib.status_counter import Status
import time
import os
import tensorflow as tf
import argparse
from lib import utils
from lib.evaluation_scripts.rank_evaluation import batch_rank_eval, batch_top_k
from lib import dataset_utils
import numpy as np
parser = argparse.ArgumentParser()


parser.add_argument("--coverage_penalty_weight", type=float, default=0.0, help="coverage_penalty")
parser.add_argument("--diverse_decoding_rate", type=float, default=0.0, help="diverse_decoding_rate")
parser.add_argument("--length_penalty_weight", type=float, default=0.0, help="length_penalty_weight")


parser.add_argument("-c", "--config_path", type=str, help="config json path")
parser.add_argument("-m", "--multi_cueword", type=str, default=None, help="multi_cueword path")
parser.add_argument("-t", "--test", type=bool, default=False, help="config json path")
parser.add_argument("-w", "--distribution", type=bool, default=False, help="config json path")
parser.add_argument("-b", "--beam", type=int, default=-1, help="beam search width")
parser.add_argument("-e", "--export_history", type=int, default=-1, help="export_history")
args = parser.parse_args()



def train():
    # Load config
    hparams = config_parser.load_and_restore_config(args.config_path, verbose=True)
    out_dir = hparams['model_path']
    eval_file = os.path.join(out_dir, 'eval_out.txt')

    status_per_steps = hparams['status_per_steps']
    status_counter = Status(status_per_steps)

    # Dataset
    dataset = dataset_utils.create_flexka3_iterator(hparams)
    if hparams.get('rank_based', False):
        model = RModel(dataset, hparams, model_helper.TRAIN)
    else:
        model = Model(dataset, hparams, model_helper.TRAIN)
    dropout = dataset['dropout']

    with tf.Session(config=model_helper.create_tensorflow_config()) as sess:
        step, epoch = model_helper.create_or_restore_a_model(out_dir, model, sess)
        dataset['init_fn'](sess)
        epoch_start_time = time.time()
        while utils.should_stop(epoch, step, hparams) is False:
            try:
                gradient,lr, _, loss, regulation_loss, step, epoch, batch_size, cue_fact, probs, kld_loss = sess.run([
                    model.grad_norm,model.learning_rate, model.update, model._knowledge_bow_loss, model.regulation_loss, model.global_step, model.epoch_step,
                    model.batch_size, dataset['cue_fact'],
                    model.classifier_scores, model.kld_loss
                ],
                    feed_dict={dropout: hparams['dropout'], model.learning_rate: hparams['learning_rate']})

                ranks, reversed_ranks, hits = batch_rank_eval(cue_fact, probs, hitAT=(1, 5, 10, 20))
                MR = np.average(ranks)
                MRR = np.average(reversed_ranks)
                hit1 = np.average(hits[0]) * 100
                hit5 = np.average(hits[1]) * 100
                hit10 = np.average(hits[2]) * 100
                hit20 = np.average(hits[3]) * 100

                # print(sess.run(model.debug))
                status_counter.add_record({'gradient':gradient,'loss': loss, 'kld': kld_loss*1000000, 'lr': lr,
                                           'MR':MR, 'MRR':MRR,
                                           'hit1':hit1, 'hit5':hit5, 'hit10':hit10, 'hit20':hit20
                                           }, step, epoch)

            except tf.errors.InvalidArgumentError as e:
                print('Found Inf or NaN global norm')
                raise e
            except tf.errors.OutOfRangeError:
                utils.print_out('epoch %d is finished,  step %d' % (epoch, step))
                sess.run([model.next_epoch])
                # Save Epoch
                model.saver.save(
                    sess,
                    os.path.join(out_dir, "seq2seq.ckpt"),
                    global_step=model.global_step)
                utils.print_out('Saved model to -> %s' % out_dir)

                # EVAL on Dev/Test Set:
                for prefix in ['valid_', 'test_']:
                    dataset['init_fn'](sess, prefix)
                    eval_loss = []
                    eval_count = []
                    eval_batch = []
                    MRs = []
                    MRRs = []
                    hit1s = []
                    hit5s = []
                    hit10s = []
                    hit20s = []

                    while True:
                        try:
                            loss, batch_size, batch_size,cue_fact, probs,kld_loss = sess.run(
                                [model._knowledge_bow_loss, model.batch_size, model.batch_size,
                                 dataset['cue_fact'], model.classifier_scores, model.kld_loss],
                                feed_dict={dropout: 0.0})
                            eval_loss.append(loss)
                            eval_batch.append(batch_size)

                            ranks, reversed_ranks, hits = batch_rank_eval(cue_fact, probs, hitAT=(1, 5, 10, 20))
                            MRs = MRs + ranks
                            MRRs = MRRs + reversed_ranks
                            hit1s = hit1s + hits[0]
                            hit5s = hit5s + hits[1]
                            hit10s = hit10s + hits[2]
                            hit20s = hit20s + hits[3]


                        except tf.errors.OutOfRangeError as e:
                            pass
                            break
                    loss = sum(eval_loss) / len(eval_loss)
                    MR = np.average(MRs)
                    MRR = np.average(MRRs)
                    hit1 = np.average(hit1s) * 100
                    hit5 = np.average(hit5s) * 100
                    hit10 = np.average(hit10s) * 100
                    hit20 = np.average(hit20s) * 100
                    KLD = kld_loss*1000000

                    if prefix == 'valid_':
                        utils.print_out('Eval on Dev: EVAL LOSS: %.4f' % (loss))
                        utils.eval_print(eval_file, 'Eval on Dev: Epoch %d Step %d EVAL LOSS: %.4f' % (epoch, step, loss))
                        utils.print_out('Eval on Dev KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (KLD,MR,MRR,hit1,hit5,hit10,hit20))
                        utils.eval_print(eval_file,
                                         'Eval on Dev KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (KLD,MR,MRR,hit1,hit5,hit10,hit20))

                        hparams['loss'].append(float(loss))
                        hparams['epochs'].append(int(step))
                        config_parser.save_config(hparams)
                        if min(hparams['loss']) - loss >= 0:
                            model.ppl_saver.save(
                                sess,
                                os.path.join(out_dir, 'min_ppl', "seq2seq.ckpt"),
                                global_step=model.global_step)
                            utils.print_out('Saved min_ppl model to -> %s' % out_dir)

                        if len(hparams['loss']) > 1:
                            if hparams['loss'][-1] > hparams['loss'][-2]:
                                hparams['learning_rate'] = hparams['learning_rate'] * hparams['learning_halve']
                                utils.eval_print(eval_file, 'Halved the learning rate to %f' % hparams['learning_rate'])
                                config_parser.save_config(hparams)
                    else:
                        utils.print_out('Eval on Test: EVAL PPL: %.4f' % (loss))
                        utils.print_out('Eval on Test KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                        KLD,MR, MRR, hit1, hit5, hit10, hit20))
                        utils.eval_print(eval_file,
                                         'Eval on Test KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                        KLD,MR, MRR, hit1, hit5, hit10, hit20))
                        utils.eval_print(eval_file,
                                         'Eval on Test: Epoch %d Step %d EVAL PPL: %.4f' % (epoch, step, loss))

                # NEXT EPOCH
                epoch_time = time.time() - epoch_start_time
                utils.print_time(epoch_time, 'Epoch Time:')
                epoch_time = time.time() - epoch_start_time
                epoch_time *= (hparams['num_train_epochs'] - epoch - 1)
                utils.print_time(epoch_time, 'Reaming Time:')
                epoch_start_time = time.time()

                dataset['init_fn'](sess)

        utils.print_out('model has been fully trained !')





def test():
    # Dataset

    hparams = config_parser.load_and_restore_config(args.config_path, verbose=True)
    if args.beam != -1:
        hparams['beam_width'] = args.beam
        utils.print_out("Reset beam_width to %d" % args.beam)
    if args.beam > 10:
        hparams['batch_size'] = hparams['batch_size'] * 30 // args.beam

    hparams['length_penalty_weight'] = args.length_penalty_weight
    hparams['diverse_decoding_rate'] = args.diverse_decoding_rate
    hparams['coverage_penalty_weight'] = args.coverage_penalty_weight

    # Dataset
    dataset = dataset_utils.create_flexka3_iterator(hparams, is_eval=True)
    if hparams.get('rank_based', False):
        model = RModel(dataset, hparams, model_helper.INFER)
    else:
        model = Model(dataset, hparams, model_helper.INFER)

    dropout = dataset['dropout']
    fact_vocab = []
    with open(hparams['fact_path'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n').split()
            #entity_in_post, ent
            items[0] = 'P:'+items[0]
            items[1] = 'E:'+items[1]
            fact_vocab.append(','.join(items))

    out_dir = os.path.join(hparams['model_path'], 'min_ppl')
    if os.path.exists(os.path.join(hparams['model_path'],'decoded')) is False:
        os.mkdir(os.path.join(hparams['model_path'],'decoded'))

    top1_position_path = os.path.join(hparams['model_path'], 'decoded', 'test.predicted_golden_fact_position_top1.txt')
    topk_position_path = os.path.join(hparams['model_path'], 'decoded', 'test.predicted_golden_fact_position_top10.txt')
    top1_output_path = os.path.join(hparams['model_path'], 'decoded', 'predicted_top1.fh0')
    top10_output_path = os.path.join(hparams['model_path'], 'decoded', 'predicted_top10.fh0')
    meta_output_path = os.path.join(hparams['model_path'], 'decoded', 'fact_prediction.txt')


    test_query_file = hparams['test_src_file']
    test_response_file = hparams['test_tgt_file']

    with open(test_query_file, 'r+', encoding='utf-8') as fin:
        queries = [x.strip('\n') for x in fin.readlines()]
    with open(test_response_file, 'r+', encoding='utf-8') as fin:
        responses = [x.strip('\n') for x in fin.readlines()]

    with tf.Session(config=model_helper.create_tensorflow_config()) as sess:
        step, epoch = model_helper.create_or_restore_a_model(out_dir, model, sess)
        dataset['init_fn'](sess,'test_')

        MRs = []
        MRRs = []
        hit1s = []
        hit5s = []
        hit10s = []
        hit20s = []

        utils.print_out('Current Epoch,Step : %s/%s, Max Epoch,Step : %s/%s' % (epoch, step, hparams['num_train_epochs'], hparams['num_train_steps']))
        case_id = 0
        with open(meta_output_path, 'w+', encoding='utf-8') as fout:
            with open(top1_position_path, 'w+', encoding='utf-8') as ftop1:
                with open(topk_position_path, 'w+', encoding='utf-8') as ftopk:
                    with open(top1_output_path, 'w+', encoding='utf-8') as fout1:
                        with open(top10_output_path, 'w+', encoding='utf-8') as foutk:
                            while True:
                                try:
                                   cue_fact, facts, probs = sess.run([
                                       dataset['cue_fact'], dataset['inputs_for_facts'],  model.classifier_scores, ],
                                        feed_dict={dropout:  0.0})
                                   topk_index, topk_labels = batch_top_k(probs, facts)


                                   ranks, reversed_ranks, hits = batch_rank_eval(cue_fact, probs,
                                                                                 hitAT=(1, 5, 10, 20))
                                   MRs = MRs + ranks
                                   MRRs = MRRs + reversed_ranks
                                   hit1s = hit1s + hits[0]
                                   hit5s = hit5s + hits[1]
                                   hit10s = hit10s + hits[2]
                                   hit20s = hit20s + hits[3]


                                   for my_index, my_label in zip(topk_index, topk_labels):
                                        ftop1.write('%s\n' % my_index[0])
                                        fout1.write('%s\n' % fact_vocab[my_label[0]].split(',')[1][2:] )
                                        for index in my_index:
                                            ftopk.write('%s\n' % index)
                                            foutk.write('%s\n' % fact_vocab[index])
                                        case_id += 1
                                except tf.errors.OutOfRangeError as e:
                                    pass
                                    break
                            MR = np.average(MRs)
                            MRR = np.average(MRRs)
                            hit1 = np.average(hit1s) * 100
                            hit5 = np.average(hit5s) * 100
                            hit10 = np.average(hit10s) * 100
                            hit20 = np.average(hit20s) * 100
                            utils.print_out('MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' %
                                            (MR, MRR, hit1, hit5, hit10, hit20))




if args.test is False:
    train()
else:
    if args.distribution:
        test_distribution()
    else:
        test()
