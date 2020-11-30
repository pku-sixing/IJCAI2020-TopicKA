from KEFU3.kefu_model3 import Model
from lib import model_helper
from lib import config_parser
from lib.status_counter import Status
from lib.evaluation_scripts.rank_evaluation import batch_rank_eval, batch_top_k
import time
import os
import tensorflow as tf
import argparse
from lib import utils
from lib import dataset_utils
import numpy as np
parser = argparse.ArgumentParser()


parser.add_argument("--coverage_penalty_weight", type=float, default=0.0, help="coverage_penalty")
parser.add_argument("--diverse_decoding_rate", type=float, default=0.0, help="diverse_decoding_rate")
parser.add_argument("--length_penalty_weight", type=float, default=0.0, help="length_penalty_weight")


parser.add_argument("-c", "--config_path", type=str, help="config json path")
parser.add_argument("--repeat", type=int, default=0,  help="config json path")
parser.add_argument("--pretrain_config_path", type=str, default=None, help="config json path")
parser.add_argument("-m", "--multi_cueword", type=str, default=None, help="multi_cueword path")
parser.add_argument("--test_cueword", type=str, default=None, help="multi_cueword path")
parser.add_argument("-t", "--test", type=bool, default=False, help="config json path")
parser.add_argument("-w", "--distribution", type=bool, default=False, help="config json path")
parser.add_argument("-b", "--beam", type=int, default=-1, help="beam search width")
parser.add_argument("-e", "--export_history", type=int, default=-1, help="export_history")
args = parser.parse_args()



def train():
    # Load config
    hparams = config_parser.load_and_restore_config(args.config_path, verbose=True)
    if  hparams.get("round_train") is True:
        round_train()
        return

    out_dir = hparams['model_path']
    eval_file = os.path.join(out_dir, 'eval_out.txt')

    status_per_steps = hparams['status_per_steps']
    status_counter = Status(status_per_steps)

    # Dataset
    dataset = dataset_utils.create_flexka3_iterator(hparams)
    model = Model(dataset, hparams, model_helper.TRAIN)
    dropout = dataset['dropout']

    with tf.Session(config=model_helper.create_tensorflow_config()) as sess:

        # reload hparams
        if args.pretrain_config_path is not None:
            pretrain_hparams = config_parser.load_and_restore_config(args.pretrain_config_path, verbose=True)
            pretrain_out_dir = os.path.join(pretrain_hparams['model_path'], 'min_ppl')
            utils.print_out("load from pretrain %s" % pretrain_out_dir)

            step, epoch = model_helper.create_or_restore_a_model(pretrain_out_dir, model, sess)
            sess.run([model.reset_epoch, model.rest_global_step])
            step, epoch =  sess.run([model.global_step, model.epoch_step])


            # EVAL on Dev/Test Set:
            for prefix in ['valid_', 'test_']:
                dataset['init_fn'](sess, prefix)
                eval_loss = []
                eval_count = []
                eval_batch = []
                while True:
                    try:
                        loss, predict_count, batch_size, batch_size = sess.run(
                            [model.train_loss, model.predict_count, model.batch_size, model.batch_size],
                            feed_dict={dropout: 0.0,
                                       model.inference_mode: bool(
                                hparams.get('flexka_use_sampled_cue_facts_in_dev', False))}, )
                        eval_loss.append(loss)
                        eval_count.append(predict_count)
                        eval_batch.append(batch_size)
                    except tf.errors.OutOfRangeError as e:
                        pass
                        break
                ppl = utils.safe_exp(sum(eval_loss) * sum(eval_batch) / len(eval_batch) / sum(eval_count))

                if prefix == 'valid_':
                    utils.print_out('Eval on Dev: EVAL PPL: %.4f' % (ppl))
                    utils.eval_print(eval_file, 'Eval on Dev: Epoch %d Step %d EVAL PPL: %.4f' % (epoch, step, ppl))
                else:
                    utils.print_out('Eval on Test: EVAL PPL: %.4f' % (ppl))
                    utils.eval_print(eval_file,
                                     'Eval on Test: Epoch %d Step %d EVAL PPL: %.4f' % (epoch, step, ppl))

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
                        loss, batch_size, batch_size, cue_fact, probs, kld_loss = sess.run(
                            [model.knowledge_bow_loss, model.batch_size, model.batch_size,
                             dataset['cue_fact'], model.classifier_scores, model.kld_loss],
                            feed_dict={dropout: 0.0,
                                 model.inference_mode: False})
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
                KLD = kld_loss * 1000000

                if prefix == 'valid_':
                    utils.print_out('Eval on Dev: EVAL LOSS: %.4f' % (loss))
                    utils.eval_print(eval_file,
                                     'Eval on Dev: Epoch %d Step %d EVAL LOSS: %.4f' % (epoch, step, loss))
                    utils.print_out(
                        'Eval on Dev KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                            KLD, MR, MRR, hit1, hit5, hit10, hit20))
                    utils.eval_print(eval_file,
                                     'Eval on Dev KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                                         KLD, MR, MRR, hit1, hit5, hit10, hit20))

                else:
                    utils.print_out('Eval on Test: EVAL PPL: %.4f' % (loss))
                    utils.print_out(
                        'Eval on Test KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                            KLD, MR, MRR, hit1, hit5, hit10, hit20))
                    utils.eval_print(eval_file,
                                     'Eval on Test KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                                         KLD, MR, MRR, hit1, hit5, hit10, hit20))
                    utils.eval_print(eval_file,
                                     'Eval on Test: Epoch %d Step %d EVAL PPL: %.4f' % (epoch, step, loss))
            utils.print_out('Loaded Pretrain Model')


        else:
            step, epoch = model_helper.create_or_restore_a_model(out_dir, model, sess)
        dataset['init_fn'](sess)
        epoch_start_time = time.time()
        while utils.should_stop(epoch, step, hparams) is False:
            try:
                clasiffier_kld_loss,teach_force_loss, kld_loss,\
                knowledge_bow_loss, word_bow_loss, lr, _, loss, step,\
                epoch, predict_count, batch_size= sess.run([
                    model.clasiffier_kld_loss, model.teach_force_loss, model.kld_loss, model.knowledge_bow_loss, model.word_bow_loss,
                    model.learning_rate, model.update, model.train_loss,
                    model.global_step, model.epoch_step,
                    model.predict_count, model.batch_size],
                    feed_dict={dropout: hparams['dropout'],
                               model.inference_mode: bool(hparams.get('flexka_use_sampled_cue_facts_in_train', False)),
                               model.learning_rate: hparams['learning_rate']},

                )

                # print(sess.run(model.debug))
                ppl = utils.safe_exp(loss * batch_size / predict_count)

                status_counter.add_record({ 'ppl': ppl, 'loss': loss,
                                            'BOW': word_bow_loss,
                                            'KBOW':knowledge_bow_loss,
                                            'kld_classifier':clasiffier_kld_loss*1000000,
                                            'kld_loss':kld_loss*1000000, 'lr': lr,
                                            'count':predict_count,
                                            }, step, epoch)
                # raise NotImplementedError()
            except tf.errors.ResourceExhaustedError as e:
                print('############### OOM Error')
                print(e)

            except tf.errors.InvalidArgumentError as e:
                print('Found Inf or NaN global norm')
                raise e

            except tf.errors.OutOfRangeError:
            # except Exception as e:
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
                    while True:
                        try:
                            loss, predict_count, batch_size, batch_size = sess.run(
                                [model.train_loss, model.predict_count, model.batch_size, model.batch_size],
                                feed_dict={dropout: 0.0, model.inference_mode: bool(hparams.get('flexka_use_sampled_cue_facts_in_dev', False))},)
                            eval_loss.append(loss)
                            eval_count.append(predict_count)
                            eval_batch.append(batch_size)
                        except tf.errors.OutOfRangeError as e:
                            pass
                            break
                    ppl = utils.safe_exp(sum(eval_loss) * sum(eval_batch) / len(eval_batch) / sum(eval_count))

                    if prefix == 'valid_':
                        utils.print_out('Eval on Dev: EVAL PPL: %.4f' % (ppl))
                        utils.eval_print(eval_file, 'Eval on Dev: Epoch %d Step %d EVAL PPL: %.4f' % (epoch, step, ppl))

                        hparams['loss'].append(float(ppl))
                        hparams['epochs'].append(int(step))
                        config_parser.save_config(hparams)

                        if min(hparams['loss']) - ppl >= 0:
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
                        utils.print_out('Eval on Test: EVAL PPL: %.4f' % (ppl))
                        utils.eval_print(eval_file,
                                         'Eval on Test: Epoch %d Step %d EVAL PPL: %.4f' % (epoch, step, ppl))

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
                            loss, batch_size, batch_size, cue_fact, probs, kld_loss = sess.run(
                                [model.knowledge_bow_loss, model.batch_size, model.batch_size,
                                 dataset['cue_fact'], model.classifier_scores, model.kld_loss],
                                feed_dict={dropout: 0.0, model.inference_mode: False})
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
                    KLD = kld_loss * 1000000

                    if prefix == 'valid_':
                        utils.print_out('Eval on Dev: EVAL LOSS: %.4f' % (loss))
                        utils.eval_print(eval_file,
                                         'Eval on Dev: Epoch %d Step %d EVAL LOSS: %.4f' % (epoch, step, loss))
                        utils.print_out(
                            'Eval on Dev KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                            KLD, MR, MRR, hit1, hit5, hit10, hit20))
                        utils.eval_print(eval_file,
                                         'Eval on Dev KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                                         KLD, MR, MRR, hit1, hit5, hit10, hit20))

                    else:
                        utils.print_out('Eval on Test: EVAL PPL: %.4f' % (loss))
                        utils.print_out(
                            'Eval on Test KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                                KLD, MR, MRR, hit1, hit5, hit10, hit20))
                        utils.eval_print(eval_file,
                                         'Eval on Test KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                                             KLD, MR, MRR, hit1, hit5, hit10, hit20))
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




def round_train():
    # Load config
    hparams = config_parser.load_and_restore_config(args.config_path, verbose=True)
    out_dir = hparams['model_path']
    eval_file = os.path.join(out_dir, 'eval_out.txt')

    status_per_steps = hparams['status_per_steps']
    status_counter_genenator = Status(status_per_steps)
    status_counter_recommender = Status(status_per_steps)


    print("Round Train")
    # Dataset
    dataset = dataset_utils.create_flexka3_iterator(hparams)
    model = Model(dataset, hparams, model_helper.TRAIN)
    dropout = dataset['dropout']

    with tf.Session(config=model_helper.create_tensorflow_config()) as sess:
        step, epoch = model_helper.create_or_restore_a_model(out_dir, model, sess)
        dataset['init_fn'](sess)
        epoch_start_time = time.time()
        while utils.should_stop(epoch, step, hparams) is False:
            try:
                if epoch % 2 == 1:
                    teach_force_loss, kld_loss,\
                     word_bow_loss, lr, _, loss, step,\
                    epoch, predict_count, batch_size= sess.run([
                        model.teach_force_loss, model.kld_loss,  model.word_bow_loss,
                        model.learning_rate, model.update_generator, model._generator_update_loss,
                        model.global_step, model.epoch_step,
                        model.predict_count, model.batch_size],
                        feed_dict={dropout: hparams['dropout'],
                                   model.inference_mode: True,
                                   model.learning_rate: hparams['learning_rate']})

                    # print(sess.run(model.debug))
                    ppl = utils.safe_exp(loss * batch_size / predict_count)

                    status_counter_genenator.add_record({ 'ppl': ppl, 'loss': loss,
                                                'BOW': word_bow_loss,
                                                'kld_loss':kld_loss*1000000, 'lr': lr,
                                                }, step, epoch)
                else:
                    cue_fact, probs, clasiffier_kld_loss,knowledge_bow_loss, lr, _, loss, step, \
                    epoch, predict_count, batch_size = sess.run([
                        dataset['cue_fact'], model.classifier_scores,
                        model.clasiffier_kld_loss, model.knowledge_bow_loss,
                        model.learning_rate, model.update_recommender, model._recommender_update_loss,
                        model.global_step, model.epoch_step,
                        model.predict_count, model.batch_size],
                        feed_dict={dropout: hparams['dropout'],
                                   model.inference_mode: True,
                                   model.learning_rate: hparams['learning_rate']})

                    ranks, reversed_ranks, hits = batch_rank_eval(cue_fact, probs, hitAT=(1, 5, 10, 20))

                    MR = np.average(ranks)
                    MRR = np.average(reversed_ranks)
                    hit1 = np.average(hits[0]) * 100
                    hit5 = np.average(hits[1]) * 100
                    hit10 = np.average(hits[2]) * 100
                    hit20 = np.average(hits[3]) * 100


                    status_counter_recommender.add_record({'loss': loss,
                                                'MR':MR,
                                                   'hit1':hit1,
                                                           'hit10':hit5,
                                               'KBOW': knowledge_bow_loss,
                                               'kld_classifier': clasiffier_kld_loss * 1000000,
                                               'lr': lr,
                                               }, step, epoch)

            except tf.errors.InvalidArgumentError as e:
                print('Found Inf or NaN global norm')
                raise e
            except tf.errors.ResourceExhaustedError as e:
                print('############### OOM Error')
                print(e)

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
                    while True:
                        try:
                            loss, predict_count, batch_size, batch_size = sess.run(
                                [model.train_loss, model.predict_count, model.batch_size, model.batch_size],
                                feed_dict={dropout: 0.0, model.inference_mode: True},)
                            eval_loss.append(loss)
                            eval_count.append(predict_count)
                            eval_batch.append(batch_size)
                        except tf.errors.OutOfRangeError as e:
                            pass
                            break
                    ppl = utils.safe_exp(sum(eval_loss) * sum(eval_batch) / len(eval_batch) / sum(eval_count))

                    if prefix == 'valid_':
                        utils.print_out('Eval on Dev: EVAL PPL: %.4f' % (ppl))
                        utils.eval_print(eval_file, 'Eval on Dev: Epoch %d Step %d EVAL PPL: %.4f' % (epoch, step, ppl))


                        hparams['epochs'].append(int(step))
                        config_parser.save_config(hparams)
                        if epoch % 2 == 1:
                            hparams['loss'].append(float(ppl))
                            if min(hparams['loss']) - ppl >= 0:
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
                            utils.print_out('Skipped Model Saving')
                    else:
                        utils.print_out('Eval on Test: EVAL PPL: %.4f' % (ppl))
                        utils.eval_print(eval_file,
                                         'Eval on Test: Epoch %d Step %d EVAL PPL: %.4f' % (epoch, step, ppl))

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
                            loss, batch_size, batch_size, cue_fact, probs, kld_loss = sess.run(
                                [model.knowledge_bow_loss, model.batch_size, model.batch_size,
                                 dataset['cue_fact'], model.classifier_scores, model.kld_loss],
                                feed_dict={dropout: 0.0, model.inference_mode: False})
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
                    KLD = kld_loss * 1000000

                    if prefix == 'valid_':
                        utils.print_out('Eval on Dev: EVAL LOSS: %.4f' % (loss))
                        utils.eval_print(eval_file,
                                         'Eval on Dev: Epoch %d Step %d EVAL LOSS: %.4f' % (epoch, step, loss))
                        utils.print_out(
                            'Eval on Dev KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                            KLD, MR, MRR, hit1, hit5, hit10, hit20))
                        utils.eval_print(eval_file,
                                         'Eval on Dev KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                                         KLD, MR, MRR, hit1, hit5, hit10, hit20))

                    else:
                        utils.print_out('Eval on Test: EVAL PPL: %.4f' % (loss))
                        utils.print_out(
                            'Eval on Test KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                                KLD, MR, MRR, hit1, hit5, hit10, hit20))
                        utils.eval_print(eval_file,
                                         'Eval on Test KLD=%.2f,MR=%.2f,MRR=%.2f,hit1=%.2f,hit5=%.2f,hit10=%.2f,hit20=%.2f' % (
                                             KLD, MR, MRR, hit1, hit5, hit10, hit20))
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



def test_distribution():

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

    entity_set = set()
    with open(hparams['entity_path'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n')
            entity_set.add(items)

    input_srcs = []
    input_src_lens = []
    with open(hparams['%ssrc_file' % 'test_'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n')
            input_srcs.append(items)
            input_src_lens.append(len(items.split()))

    out_dir = os.path.join(hparams['model_path'], 'min_ppl')
    if os.path.exists(os.path.join(hparams['model_path'],'decoded')) is False:
        os.mkdir(os.path.join(hparams['model_path'],'decoded'))
    if os.path.exists(os.path.join(hparams['model_path'], 'decoded','fact_attention')) is False:
        os.mkdir(os.path.join(hparams['model_path'],'decoded','fact_attention'))

    assert hparams['beam_width'] == 1 # 还不能搜索其他
    config_id = 'B%s_L%.1f_D%.1f_C%.1f' % (hparams['beam_width'] , args.length_penalty_weight, args.diverse_decoding_rate, args.coverage_penalty_weight)

    beam_out_file_path = os.path.join(hparams['model_path'], 'decoded', 'ClueFactDistribution__%s.txt' % config_id)
    beam_out_file_path4 = os.path.join(hparams['model_path'], 'decoded', 'PostClueFactDistribution__%s.txt' % config_id)
    beam_out_file_path2 = os.path.join(hparams['model_path'], 'decoded', 'DynamicFactDistribution__%s.txt' % config_id)
    beam_out_file_path3 = os.path.join(hparams['model_path'], 'decoded', 'Dynamic2FactDistribution__%s.txt' % config_id)
    top1_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_top1.txt' % config_id)

    with open(top1_out_file_path,'r+',encoding='utf-8') as fin:
        lines = fin.readlines()
        token_lengths = [len(x.strip('\r\n').split()) for x in lines]

    with tf.Session(config=model_helper.create_tensorflow_config()) as sess:
        step, epoch = model_helper.create_or_restore_a_model(out_dir, model, sess)
        dataset['init_fn'](sess,'test_')

        utils.print_out('Current Epoch,Step : %s/%s, Max Epoch,Step : %s/%s' % (epoch, step, hparams['num_train_epochs'], hparams['num_train_steps']))
        case_id = 0
        with open(beam_out_file_path, 'w+', encoding='utf-8') as fout:
            # with open(beam_out_file_path2, 'w+', encoding='utf-8') as fout2:
            #     with open(beam_out_file_path3, 'w+', encoding='utf-8') as fout3:
                    with open(beam_out_file_path4, 'w+', encoding='utf-8') as fout4:
                        while True:
                            try:
                                distribution, distribution_post, facts, lengths = sess.run( #fact_alignments
                                    [model.knowledge_distribution, model.post_knowledge_distribution, dataset['inputs_for_facts'], dataset['lengths_for_facts']], # model.fact_alignments
                                    feed_dict={dropout: 0.0, model.inference_mode: bool(hparams.get('flexka_use_sampled_cue_facts_in_test',False))})
                                # distribution2 = np.mean(fact_alignments, 0)
                                # distribution3 = []
                                # for batch_id in range(np.shape(fact_alignments)[1]):
                                #     distribution3.append(np.mean(fact_alignments[0:token_lengths[case_id+batch_id],batch_id,:], 0))
                                # time,batch,fact_len
                                distribution2 = distribution3 =distribution
                                for digits, digits2, digits3, digits4, fact_len in zip(distribution,distribution2, distribution3, distribution_post, lengths):
                                    try:
                                        fout.write('%d,' % fact_len)
                                        fout.write(','.join([str(x) for x in digits]))
                                        fout.write('\n')

                                        fout4.write('%d,' % fact_len)
                                        fout4.write(','.join([str(x) for x in digits4]))
                                        fout4.write('\n')

                                        # fout2.write('%d,' % fact_len)
                                        # fout2.write(','.join([str(x) for x in digits2]))
                                        # fout2.write('\n')
                                        #
                                        # fout3.write('%d,' % fact_len)
                                        # fout3.write(','.join([str(x) for x in digits3]))
                                        # fout3.write('\n')
                                    except Exception as e:
                                        pass

                                    case_id += 1
                            except tf.errors.OutOfRangeError as e:
                                pass
                                break



def test():
    if args.repeat > 0:
        repeat_test(args.repeat)
        return

    hparams = config_parser.load_and_restore_config(args.config_path, verbose=True)
    if args.beam != -1:
        hparams['beam_width'] = args.beam
        utils.print_out("Reset beam_width to %d" % args.beam)
    if args.test_cueword is not None:
        hparams['test_cue_fact_file'] = args.test_cueword
        utils.print_out("Reset test_cueword to %s" % args.test_cueword)

    if args.beam > 10:
        hparams['batch_size'] = hparams['batch_size'] * 30 // args.beam

    hparams['length_penalty_weight'] = args.length_penalty_weight
    hparams['diverse_decoding_rate'] = args.diverse_decoding_rate
    hparams['coverage_penalty_weight'] = args.coverage_penalty_weight

    # Dataset
    dataset = dataset_utils.create_flexka3_iterator(hparams, is_eval=True)
    model = Model(dataset, hparams, model_helper.INFER, force_argmax=True)
    dropout = dataset['dropout']
    fact_vocab = []
    with open(hparams['fact_path'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n').split()
            #entity_in_post, ent
            items[0] = 'P:'+items[0]
            items[1] = 'E:'+items[1]
            fact_vocab.append(','.join(items))

    entity_set = set()
    with open(hparams['entity_path'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n')
            entity_set.add(items)

    input_srcs = []
    input_src_lens = []
    with open(hparams['%ssrc_file' % 'test_'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n')
            input_srcs.append(items)
            input_src_lens.append(len(items.split()))

    out_dir = os.path.join(hparams['model_path'], 'min_ppl')
    if os.path.exists(os.path.join(hparams['model_path'], 'decoded')) is False:
        os.mkdir(os.path.join(hparams['model_path'], 'decoded'))
    if os.path.exists(os.path.join(hparams['model_path'], 'decoded', 'fact_attention')) is False:
        os.mkdir(os.path.join(hparams['model_path'],'decoded','fact_attention'))
    fact_attention_path = os.path.join(hparams['model_path'],'decoded','fact_attention')






    config_id = 'B%s_L%.1f_D%.1f_C%.1f' % (hparams['beam_width'] , args.length_penalty_weight, args.diverse_decoding_rate, args.coverage_penalty_weight)

    beam_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s.txt' % config_id)
    top1_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_top1.txt' % config_id)
    topk_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_topk.txt' % config_id)

    test_query_file = hparams['test_src_file']
    test_response_file = hparams['test_tgt_file']

    with open(test_query_file, 'r+', encoding='utf-8') as fin:
        queries = [x.strip('\n') for x in fin.readlines()]
    with open(test_response_file, 'r+', encoding='utf-8') as fin:
        responses = [x.strip('\n') for x in fin.readlines()]

    with tf.Session(config=model_helper.create_tensorflow_config()) as sess:
        step, epoch = model_helper.create_or_restore_a_model(out_dir, model, sess)
        dataset['init_fn'](sess,'test_')

        utils.print_out('Current Epoch,Step : %s/%s, Max Epoch,Step : %s/%s' % (epoch, step, hparams['num_train_epochs'], hparams['num_train_steps']))
        case_id = 0
        with open(beam_out_file_path, 'w+', encoding='utf-8') as fout:
            with open(top1_out_file_path, 'w+', encoding='utf-8') as ftop1:
                with open(topk_out_file_path, 'w+', encoding='utf-8') as ftopk:
                    while True:
                        try:
                            cue_facts, model_selector, facts, lengts_for_facts, src_ids, sample_ids, probs, scores = sess.run(
                                [model.cue_fact_relative_idx,
                                    model.mode_selector,   dataset['inputs_for_facts'], dataset['lengths_for_facts'],
                                 dataset['inputs_for_encoder'], model.sampled_id, model.logits, model.scores],
                                feed_dict={dropout: 0.0, model.inference_mode: bool(hparams.get('flexka_use_sampled_cue_facts_in_test',False))})
                            # print(()

                            num_responses_per_query = sample_ids.shape[0]
                            num_cases = sample_ids.shape[1]
                            for sent_id in range(num_cases):
                                fout.write('#Case : %d\n' % case_id)
                                fout.write('\tquery:\t%s\n' % queries[case_id])
                                fout.write('\tresponse:\t%s\n' % responses[case_id])
                                cue_id = cue_facts[sent_id][0]
                                fact_id = facts[sent_id][cue_id]
                                fout.write('\tfact:\t%s\n' % fact_vocab[fact_id])

                                if hparams['beam_width'] == 1 and args.export_history == 1 and hparams.get('fusion_encoder', True):
                                    input_src = input_srcs[case_id].split()
                                    for i in range(len(input_src)):
                                        if input_src[i] in entity_set:
                                            input_src[i] = input_src[i].upper()

                                if hparams['beam_width'] == 1 and hparams.get('kefu_decoder', False) and args.export_history==1:
                                    input_tags = [ '(%s)' % ','.join(fact_vocab[x].split(',')[2:]) for x in facts[sent_id,:] ]
                                    translations, score = model_helper.get_translation(sample_ids[0], scores[0],
                                                                                       sent_id, '</s>')
                                    # Translate to None ENT
                                    # for i, translation in enumerate(translations):
                                    new_translation = []
                                    for pid, token in enumerate(translations.split()):
                                        if token[:len('$ENT_')] == '$ENT_':
                                            relative_fact_id = int(token[len('$ENT_'):])
                                            fact = fact_vocab[facts[sent_id,relative_fact_id]]
                                            entity_in_response = fact.split(',')[1]
                                            new_translation.append('$'+entity_in_response)
                                        elif token[:len('$CP_')] == '$CP_':
                                            position = int(token[len('$CP_'):])
                                            new_translation.append('$C:'+input_srcs[case_id].split()[position])
                                        else:
                                            new_translation.append(token)
                                    translations = ' '.join(new_translation)


                                    input_src = input_srcs[case_id]

                                for beam_id in range(num_responses_per_query):
                                     translations, score = model_helper.get_translation(sample_ids[beam_id], scores[beam_id], sent_id, '</s>')
                                     new_translation = []
                                     for pid, token in enumerate(translations.split()):
                                         if token[:len('$ENT_')] == '$ENT_':
                                             relative_fact_id = int(token[len('$ENT_'):])
                                             fact = fact_vocab[facts[sent_id, relative_fact_id]]
                                             entity_in_response = fact.split(',')[1]
                                             new_translation.append('$' + entity_in_response)
                                         elif token[:len('$CP_')] == '$CP_':
                                             position = int(token[len('$CP_'):])
                                             new_translation.append('$C:' + input_srcs[case_id].split()[position])
                                         else:
                                             new_translation.append(token)
                                     translations = ' '.join(new_translation)
                                     fout.write('\tBeam%d\t%.4f\t%s\n' % (beam_id, score, translations))

                                     if beam_id == 0:
                                         ftop1.write('%s\n' % (translations.replace('#','').replace('$R:','').replace('$C:','').replace('$E:','')))
                                     ftopk.write('%s\n' % (
                                     translations.replace('#', '').replace('$R:', '').replace('$C:', '').replace('$E:','')))
                                case_id += 1
                        except tf.errors.OutOfRangeError as e:
                            pass
                            break

def repeat_test(repeat_times):
    hparams = config_parser.load_and_restore_config(args.config_path, verbose=True)
    if args.beam != -1:
        hparams['beam_width'] = args.beam
        utils.print_out("Reset beam_width to %d" % args.beam)
    if args.test_cueword is not None:
        hparams['test_cue_fact_file'] = args.test_cueword
        utils.print_out("Reset test_cueword to %s" % args.test_cueword)

    if args.beam > 10:
        hparams['batch_size'] = hparams['batch_size'] * 30 // args.beam

    hparams['length_penalty_weight'] = args.length_penalty_weight
    hparams['diverse_decoding_rate'] = args.diverse_decoding_rate
    hparams['coverage_penalty_weight'] = args.coverage_penalty_weight

    # Dataset
    dataset = dataset_utils.create_flexka3_iterator(hparams, is_eval=True)
    model = Model(dataset, hparams, model_helper.INFER, force_argmax=False)
    dropout = dataset['dropout']
    fact_vocab = []
    with open(hparams['fact_path'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n').split()
            #entity_in_post, ent
            items[0] = 'P:'+items[0]
            items[1] = 'E:'+items[1]
            fact_vocab.append(','.join(items))

    entity_set = set()
    with open(hparams['entity_path'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n')
            entity_set.add(items)

    input_srcs = []
    input_src_lens = []
    with open(hparams['%ssrc_file' % 'test_'], encoding='utf-8') as fin:
        for line in fin.readlines():
            items = line.strip('\n')
            input_srcs.append(items)
            input_src_lens.append(len(items.split()))

    out_dir = os.path.join(hparams['model_path'], 'min_ppl')
    if os.path.exists(os.path.join(hparams['model_path'], 'decoded')) is False:
        os.mkdir(os.path.join(hparams['model_path'], 'decoded'))
    if os.path.exists(os.path.join(hparams['model_path'], 'decoded', 'fact_attention')) is False:
        os.mkdir(os.path.join(hparams['model_path'],'decoded','fact_attention'))



    config_id = 'B%s_L%.1f_D%.1f_C%.1f' % (repeat_times , args.length_penalty_weight, args.diverse_decoding_rate, args.coverage_penalty_weight)
    beam_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s.txt' % config_id)
    top1_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_top1.txt' % config_id)
    topk_out_file_path = os.path.join(hparams['model_path'], 'decoded', '%s_topk.txt' % config_id)

    test_query_file = hparams['test_src_file']
    test_response_file = hparams['test_tgt_file']

    with open(test_query_file, 'r+', encoding='utf-8') as fin:
        queries = [x.strip('\n') for x in fin.readlines()]
    with open(test_response_file, 'r+', encoding='utf-8') as fin:
        responses = [x.strip('\n') for x in fin.readlines()]

    decoded_raws = []
    decoded_cue_ids = []
    with tf.Session(config=model_helper.create_tensorflow_config()) as sess:
        step, epoch = model_helper.create_or_restore_a_model(out_dir, model, sess)
        for repeat_time in range(repeat_times):
            decoded_raw = []
            decoded_cue_id = []
            dataset['init_fn'](sess,'test_')
            utils.print_out('Current Epoch,Step : %s/%s, Max Epoch,Step : %s/%s' % (epoch, step, hparams['num_train_epochs'], hparams['num_train_steps']))
            case_id = 0
            while True:
                try:
                    cue_facts, model_selector, facts, lengts_for_facts, src_ids, sample_ids, probs, scores = sess.run(
                        [model.cue_fact_relative_idx,
                            model.mode_selector,   dataset['inputs_for_facts'], dataset['lengths_for_facts'],
                         dataset['inputs_for_encoder'], model.sampled_id, model.logits, model.scores],
                        feed_dict={dropout: 0.0, model.inference_mode: bool(hparams.get('flexka_use_sampled_cue_facts_in_test',False))})
                    # print(()

                    num_responses_per_query = sample_ids.shape[0]
                    num_cases = sample_ids.shape[1]
                    for sent_id in range(num_cases):
                        cue_id = cue_facts[sent_id][0]
                        fact_id = facts[sent_id][cue_id]
                        decoded_cue_id.append(fact_id)
                        if hparams['beam_width'] == 1 and args.export_history == 1 and hparams.get('fusion_encoder', True):
                            input_src = input_srcs[case_id].split()
                            for i in range(len(input_src)):
                                if input_src[i] in entity_set:
                                    input_src[i] = input_src[i].upper()

                        for beam_id in range(num_responses_per_query):
                             translations, score = model_helper.get_translation(sample_ids[beam_id], scores[beam_id], sent_id, '</s>')
                             new_translation = []
                             for pid, token in enumerate(translations.split()):
                                 if token[:len('$ENT_')] == '$ENT_':
                                     relative_fact_id = int(token[len('$ENT_'):])
                                     fact = fact_vocab[facts[sent_id, relative_fact_id]]
                                     entity_in_response = fact.split(',')[1]
                                     new_translation.append('$' + entity_in_response)
                                 elif token[:len('$CP_')] == '$CP_':
                                     position = int(token[len('$CP_'):])
                                     new_translation.append('$C:' + input_srcs[case_id].split()[position])
                                 else:
                                     new_translation.append(token)
                             translations = ' '.join(new_translation)
                             decoded_raw.append(translations)
                        case_id += 1
                except tf.errors.OutOfRangeError as e:
                    pass
                    break
            decoded_raws.append(decoded_raw)
            decoded_cue_ids.append(decoded_cue_id)

            with open(beam_out_file_path, 'w+', encoding='utf-8') as fout:
                with open(top1_out_file_path, 'w+', encoding='utf-8') as ftop1:
                    with open(topk_out_file_path, 'w+', encoding='utf-8') as ftopk:
                        k = len(decoded_raws)
                        n = len(decoded_raws[0])

                        for case_id in range(n):
                            fout.write('#Case : %d\n' % case_id)
                            fout.write('\tquery:\t%s\n' % queries[case_id])
                            fout.write('\tresponse:\t%s\n' % responses[case_id])
                            for i in range(k):
                                fout.write('\tFact:\t%s\n' % fact_vocab[decoded_cue_ids[i][case_id]])
                                fout.write('\tGeneration:\t%s\n' % decoded_raws[i][case_id])
                                translations =  decoded_raws[i][case_id].replace('#', '').replace('$R:', '').replace('$C:', '').replace('$E:', '')
                                if i == 0:
                                    ftop1.write('%s\n' % (translations))
                                ftopk.write('%s\n' % (translations))








if args.test is False:
    train()
else:
    if args.distribution:
        test_distribution()
    else:
        test()
