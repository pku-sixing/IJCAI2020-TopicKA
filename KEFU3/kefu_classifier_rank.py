import tensorflow as tf
from lib import model_helper
from lib import utils

class Model:

    def __init__(self, inputs, hparams, mode):
        self.word_bow_loss = tf.constant(0.0)
        vocab_size = hparams['tgt_vocab_size']

        self.mode = mode
        self.dropout = inputs['dropout']
        self._word2entity = inputs['word2entity']
        self._entity2word = inputs['entity2word']
        self._fact_distribution = hparams.get('fact_distribution', False)

        # 鉴别是词，还是Entity Word 还是什么）
        id2type = tf.constant([0] * vocab_size + [1] * hparams['copy_token_nums'] + [2] * hparams['entity_token_nums'],
                              dtype=tf.int32)
        self._embedding_id2type = tf.one_hot(id2type, depth=3, dtype=tf.float32)
        self._inputs_for_encoder = inputs['inputs_for_encoder']
        self._inputs_for_decoder = inputs['inputs_for_decoder']
        self._outputs_for_decoder = inputs['outputs_for_decoder']
        self._outputs_type_for_decoder = tf.nn.embedding_lookup(id2type, self._outputs_for_decoder)
        self.batch_size = tf.to_float(tf.shape(self._inputs_for_encoder)[0])

        # 都是相对位置
        self._cue_fact_idx = tf.squeeze(inputs['cue_fact'], -1)
        self._neg_fact_idx = tf.squeeze(inputs['neg_fact'], -1)

        self._lengths_for_fact_candidate = inputs['lengths_for_facts']

        self._lengths_for_encoder = inputs['lengths_for_encoder']
        self._lengths_for_decoder = inputs['lengths_for_decoder']

        self._embedding_vocab = inputs['embedding_vocab']
        self._embedding_entity = inputs['embedding_entity']
        self._embedding_fact = inputs['embedding_fact']

        self._fact_entity_in_response = inputs['fact_entity_in_response']
        self._fact_entity_in_post = inputs['fact_entity_in_post']
        self._fact_candidate = inputs['inputs_for_facts']

        self._max_fact_num = tf.shape(self._fact_candidate)[-1]

        # => [batch, fact_num]
        zero_mask = tf.one_hot(tf.zeros([self.batch_size], dtype=tf.int32)
                               , depth=vocab_size,
                               on_value=0.0, off_value=1.0,
                               dtype=tf.float32)
        word_bow = tf.reduce_max(tf.one_hot(self._outputs_for_decoder, depth=vocab_size,
                                            on_value=1.0, off_value=0.0,
                                            dtype=tf.float32), 1)
        self._golden_word_bow = tf.minimum(word_bow, zero_mask)

        # => [batch, fact_num]
        zero_mask = tf.one_hot(tf.zeros([self.batch_size], dtype=tf.int32)
                               , depth=self._max_fact_num,
                               on_value=0.0, off_value=1.0,
                               dtype=tf.float32)
        golden_fact_bow = tf.reduce_max(tf.one_hot(inputs['cue_fact'], depth=self._max_fact_num,
                                                   on_value=1.0, off_value=0.0,
                                                   dtype=tf.float32), 1)
        self._golden_fact_bow = tf.minimum(golden_fact_bow, zero_mask)

        neg_fact_bow = tf.reduce_max(tf.one_hot(inputs['neg_fact'], depth=self._max_fact_num,
                                                   on_value=1.0, off_value=0.0,
                                                   dtype=tf.float32), 1)
        self._neg_fact_bow = tf.minimum(neg_fact_bow, zero_mask)




        # 屏蔽第一个




        if hparams.get('add_word_embedding_to_fact', False):  # TODO Check 是否有必要保留
            response_words = tf.nn.embedding_lookup(self._entity2word, self._fact_entity_in_response)
            response_embedding = tf.nn.embedding_lookup(self._embedding_vocab, response_words)

            post_words = tf.nn.embedding_lookup(self._entity2word, self._fact_entity_in_post)
            post_embedding = tf.nn.embedding_lookup(self._embedding_vocab, post_words)

            self._embedding_fact = tf.concat([self._embedding_fact, post_embedding, response_embedding], axis=-1)



        # 绝对位置
        self._cue_fact_abs_idx = tf.squeeze(tf.batch_gather(inputs['inputs_for_facts'], inputs['cue_fact']), -1)
        self._cue_fact_embedding = tf.nn.embedding_lookup(self._embedding_fact, self._cue_fact_abs_idx)

        # => [batch, fact_num]
        zero_mask = tf.one_hot(tf.zeros([self.batch_size], dtype=tf.int32)
                               , depth=self._max_fact_num,
                               on_value=0.0, off_value=1.0,
                               dtype=tf.float32)
        golden_fact_bow = tf.reduce_max(tf.one_hot(inputs['cue_fact'], depth=self._max_fact_num,
                                                   on_value=1.0, off_value=0.0,
                                                   dtype=tf.float32), 1)
        self._golden_fact_bow = tf.minimum(golden_fact_bow, zero_mask)

        self._fact_candidate_embedding = tf.nn.embedding_lookup(self._embedding_fact, self._fact_candidate)

        self.fact_projection = tf.layers.dense(self._fact_candidate_embedding, units=300,
                                               activation=tf.nn.elu,
                                               name='dynamic_fact_projection')


        # inputs
        self._input_embeddings_for_encoder = tf.nn.embedding_lookup(self._embedding_vocab, self._inputs_for_encoder)
        self._input_entities_for_encoder = inputs['entity_inputs_for_encoder']
        self._input_entity_embeddings_for_encoder = tf.nn.embedding_lookup(self._embedding_entity,
                                                                           self._input_entities_for_encoder)

        self._input_embeddings_for_decoder = tf.nn.embedding_lookup(self._embedding_vocab, self._inputs_for_decoder)
        self._input_entities_for_decoder = inputs['entity_inputs_for_decoder']
        self._input_entity_embeddings_for_decoder = tf.nn.embedding_lookup(self._embedding_entity,
                                                                           self._input_entities_for_decoder)

        self._projection_layer = tf.layers.Dense(vocab_size, use_bias=False)
        self.predict_count = tf.reduce_sum(self._lengths_for_decoder)

        self.src_vocab_table = inputs['src_vocab_table']
        self.tgt_vocab_table = inputs['tgt_vocab_table']
        self.reverse_target_vocab_table = inputs['reverse_target_vocab_table']

        self.hparams = hparams

        self.global_step = tf.Variable(0, trainable=False)
        self.epoch_step = tf.Variable(0, trainable=False)
        self.next_epoch = tf.assign_add(self.epoch_step, 1)

        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
        # warm-up
        self.learning_rate = model_helper.get_learning_rate_warmup(self.learning_rate, self.global_step, hparams)
        # decay
        self.learning_rate = model_helper.get_learning_rate_decay(self.learning_rate, self.global_step, hparams)

        self.create_model()
        if self.mode == model_helper.TRAIN:
            self.create_update_op()

        # Saver
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=2)
        self.ppl_saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=hparams['num_keep_ckpts'])

    def create_model(self, name='flexka'):

        def safe_log(y):
            return tf.log(tf.clip_by_value(y, 1e-9, tf.reduce_max(y)))

        hparams = self.hparams
        with tf.variable_scope(name) as scope:
            encoder_outputs, encoder_states = self.create_encoder(self._input_embeddings_for_encoder,
                                                                  self._input_entity_embeddings_for_encoder,
                                                                  self._lengths_for_encoder,
                                                                  )

            self.kld_loss = tf.constant(0.0)
            self.knowledge_distribution = None
            self.knowledge_fusion = None
            self.knowledge_bow_loss = tf.constant(0.0)



            maximium_candidate_num = tf.shape(self._fact_candidate_embedding)[1]

            fact_seq_mask = tf.sequence_mask(self._lengths_for_fact_candidate, dtype=tf.float32)
            unk_mask = tf.sequence_mask(tf.ones_like(self._lengths_for_fact_candidate), maxlen=maximium_candidate_num,
                                        dtype=tf.float32)
            fact_mask = (1.0 - fact_seq_mask) * -1e10 + unk_mask * -1e10

            fact_embedding_projection = self.fact_projection
            classifier_inputs = tf.concat(encoder_states, -1)
            classifier_inputs = tf.nn.dropout(classifier_inputs, keep_prob=1.0 - self.dropout)
            classifier_projection = tf.layers.dense(classifier_inputs, units=300, activation=tf.nn.elu,
                                                    use_bias=True, name='classifier_inputs')
            classifier_projection = tf.expand_dims(classifier_projection, 1)
            classifier_projection = tf.tile(classifier_projection, [1, maximium_candidate_num, 1])
            score_input = tf.concat([classifier_projection, fact_embedding_projection], -1)
            score_input = tf.nn.dropout(score_input, keep_prob=1.0 - self.dropout)
            
            classifier_scores = tf.layers.dense(score_input, units=1, activation=tf.nn.sigmoid, name='score_estimator')
            classifier_scores = tf.squeeze(classifier_scores)
            self.classifier_scores = classifier_scores
            if self.mode == model_helper.TRAIN:

                pos_scores =  tf.reduce_sum(self._golden_fact_bow * classifier_scores, -1)
                neg_scores =  tf.reduce_sum(self._neg_fact_bow * classifier_scores, -1)
                knowledge_bow_loss = tf.maximum(0.0,  0.3 - pos_scores + neg_scores)
                self._knowledge_bow_loss = tf.reduce_sum(knowledge_bow_loss) / self.batch_size
                self._train_update_loss = self._knowledge_bow_loss
                self._knowledge_bow_loss *= 100
                self.regulation_loss = tf.constant(0.0)

        # Print vars
        utils.print_out('-------------Trainable Variables------------------')
        for var in tf.trainable_variables():
            utils.print_out(var)

    def create_update_op(self):
        hparams = self.hparams
        # Optimizer
        if hparams['optimizer'] == "sgd":
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif hparams['optimizer'] == "adam":
            opt = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError("Unknown optimizer type %s" % hparams.optimizer)

        params = tf.trainable_variables()

        gradients = tf.gradients(
            self._train_update_loss,
            params,
            colocate_gradients_with_ops=hparams['colocate_gradients_with_ops'])

        clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
            gradients, max_gradient_norm=hparams['max_gradient_norm'], safe_clip=hparams['safe_clip'])
        self.grad_norm_summary = grad_norm_summary
        self.grad_norm = grad_norm

        self.update = opt.apply_gradients(
            zip(clipped_grads, params), global_step=self.global_step)

    def create_encoder(self, seq_inputs, entity_inputs, lengths, name='encoder'):
        """

        :param inputs:  [batch,time,dimension]
        :param lengths:  [batch]
        :param hparams: hparams
        :return:
        """
        hparams = self.hparams
        mode = self.mode
        num_layers = hparams['encoder_num_layers']
        cell_type = hparams['cell_type']
        num_units = hparams['num_units']
        forget_bias = hparams['forget_bias']
        embed_dim = hparams['embed_dim']
        dropout = self.dropout

        with tf.variable_scope(name) as scope:
            inputs_for_std = seq_inputs
            inputs_for_fact = entity_inputs
            inputs = tf.concat([inputs_for_std, inputs_for_fact], axis=-1)

            # Crate KEFU Encoder RNN Cells
            def create_kefu_cell(name):
                cell_list = [model_helper.create_cell(cell_type, num_units, forget_bias, dropout, mode) for x in
                             range(2)]
                cell_fw = tf.contrib.rnn.MultiRNNCell(cell_list)
                return cell_fw

            with tf.variable_scope('Knowledge_RNN'):
                cell_fw = create_kefu_cell('KEFU_FW')
                cell_bw = create_kefu_cell('KEFU_BW')

                utils.print_out('Creating bi_directional RNN Encoder, num_layers=%s, cell_type=%s, num_units=%d' %
                                (num_layers, cell_type, num_units))

                bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    inputs,
                    dtype=tf.float32,
                    sequence_length=lengths,
                    time_major=False,
                    swap_memory=True)
                encoder_outputs = tf.concat(bi_encoder_outputs, -1)
                # 级联最后一层
                encoder_state = [tf.concat(x, -1) for x in bi_encoder_state]

            return encoder_outputs, encoder_state

    def _prepare_beam_search_decoder_inputs(
            self, beam_width, memory, source_sequence_length, encoder_state):
        memory = tf.contrib.seq2seq.tile_batch(
            memory, multiplier=beam_width)
        source_sequence_length = tf.contrib.seq2seq.tile_batch(
            source_sequence_length, multiplier=beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)
        batch_size = self.batch_size * beam_width
        return memory, source_sequence_length, encoder_state, batch_size
