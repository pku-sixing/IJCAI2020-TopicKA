import tensorflow as tf
from lib import utils
from lib import vocab_utils
import collections

TRAIN = tf.contrib.learn.ModeKeys.TRAIN
EVAL = tf.contrib.learn.ModeKeys.EVAL
INFER = tf.contrib.learn.ModeKeys.INFER


def create_tensorflow_config():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    return config


def create_vocab_from_file(src_vocab_file, tgt_vocab_file, is_share):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(src_vocab_file, tgt_vocab_file, is_share)
    return src_vocab_table, tgt_vocab_table


def create_cell(unit_type, num_units, forget_bias, dropout, mode,
                 residual_connection=False, device_str=None, residual_fn=None):

  # dropout (= 1 - keep_prob) is set to 0 during eval and infer
  dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

  # Cell Type
  if unit_type == "lstm":
    single_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units,
        forget_bias=forget_bias)
  elif unit_type == "gru":
    single_cell = tf.contrib.rnn.GRUCell(num_units)
  elif unit_type == "layer_norm_lstm":
    utils.print_out("  Layer Normalized LSTM, forget_bias=%g" % forget_bias)
    single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        num_units,
        forget_bias=forget_bias,
        layer_norm=True)
  elif unit_type == "nas":
    single_cell = tf.contrib.rnn.NASCell(num_units)
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)


  single_cell = tf.contrib.rnn.DropoutWrapper(
    cell=single_cell, input_keep_prob=(1.0 - dropout))

  # Residual
  if residual_connection:
    single_cell = tf.contrib.rnn.ResidualWrapper(
        single_cell, residual_fn=residual_fn)
    utils.print_out("  %s" % type(single_cell).__name__)

  # Device Wrapper
  if device_str:
    single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
    utils.print_out("  %s, device=%s" %
                    (type(single_cell).__name__, device_str))

  return single_cell

def get_learning_rate_warmup(learning_rate, global_step, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams['warmup_steps']
    warmup_scheme = hparams['warmup_scheme']
    utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                    (hparams['learning_rate'], warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.math.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(
          tf.cast(warmup_steps - global_step, tf.float32))
    else:
      raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        global_step < warmup_steps,
        lambda: inv_decay * learning_rate,
        lambda: learning_rate,
        name="learning_rate_warump_cond")

def get_decay_info(hparams):
    """Return decay info based on decay_scheme."""
    decay_scheme = hparams['decay_scheme']
    num_train_steps = hparams['num_train_steps']

    if decay_scheme in ["luong5", "luong10", "luong234"]:
      decay_factor = 0.5
      if decay_scheme == "luong5":
        start_decay_step = int(num_train_steps / 2)
        decay_times = 5
      elif decay_scheme == "luong10":
        start_decay_step = int(num_train_steps / 2)
        decay_times = 10
      elif decay_scheme == "luong234":
        start_decay_step = int(num_train_steps * 2 / 3)
        decay_times = 4
      remain_steps = num_train_steps - start_decay_step
      decay_steps = int(remain_steps / decay_times)
    elif not decay_scheme:  # no decay
      start_decay_step = num_train_steps
      decay_steps = 0
      decay_factor = 1.0
    elif decay_scheme:
      raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
    return start_decay_step, decay_steps, decay_factor

def get_learning_rate_decay(learning_rate,global_step, hparams):
    """Get learning rate decay."""
    decay_scheme = hparams['decay_scheme']
    start_decay_step, decay_steps, decay_factor = get_decay_info(hparams)
    utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                    "decay_factor %g" % (decay_scheme,
                                         start_decay_step,
                                         decay_steps,
                                         decay_factor))

    return tf.cond(
        global_step < start_decay_step,
        lambda: learning_rate,
        lambda: tf.compat.v1.train.exponential_decay(
            learning_rate,
            (global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")

def gradient_clip(gradients, max_gradient_norm,safe_clip):
  """Clipping gradients of a model."""
  if safe_clip:
      utils.print_out('Enable Safe Clip')
      safe_value = max_gradient_norm
      gradients = [tf.clip_by_value(x, -safe_value, safe_value) for x in gradients]
      gradient_norm = tf.reduce_mean(gradients[0])
      # clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      #     gradients, max_gradient_norm)
      gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
      gradient_norm_summary.append(
          tf.summary.scalar("clipped_gradient", gradient_norm))
      return gradients, gradient_norm_summary, gradient_norm

  else:
      clipped_gradients, gradient_norm = tf.clip_by_global_norm(
          gradients, max_gradient_norm)
      gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
      gradient_norm_summary.append(
          tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

      return clipped_gradients, gradient_norm_summary, gradient_norm

def print_variables_in_ckpt(ckpt_path):
  """Print a list of variables in a checkpoint together with their shapes."""
  utils.print_out("# Variables in ckpt %s" % ckpt_path)
  reader = tf.train.NewCheckpointReader(ckpt_path)
  variable_map = reader.get_variable_to_shape_map()
  for key in sorted(variable_map.keys()):
    utils.print_out("  %s: %s" % (key, variable_map[key]))


def format_text(words):
  """Convert a sequence words into sentence."""
  if (not hasattr(words, "__len__") and  # for numpy array
      not isinstance(words, collections.Iterable)):
    words = [words]
  return b" ".join(words)

def get_translation(nmt_outputs, nmt_scores, sent_id, tgt_eos, subword_option=None):
  """Given batch decoding outputs, select a sentence and turn to text."""
  if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
  # Select a sentence
  output = nmt_outputs[sent_id, :].tolist()
  score = min(nmt_scores[sent_id, :])

  # If there is an eos symbol in outputs, cut them at that point.
  if tgt_eos and tgt_eos in output:
    output = output[:output.index(tgt_eos)]
  translation = format_text(output).decode('utf-8')
  return translation, score

def create_or_restore_a_model(out_dir, model, sess):
    latest_ckpt = tf.train.latest_checkpoint(out_dir)
    if latest_ckpt:
        try:
            print('Try to load from %s' % latest_ckpt)
            model.saver.restore(sess, latest_ckpt)
        except tf.errors.NotFoundError as e:
            utils.print_out("Can't load checkpoint")
            print_variables_in_ckpt(latest_ckpt)
            utils.print_out("%s" % str(e))
            raise e

        sess.run(tf.tables_initializer())
        utils.print_out(
            "  loaded model parameters from %s" % (latest_ckpt))

        step, epoch = sess.run([model.global_step, model.epoch_step])
    else:
        init_op = tf.random_uniform_initializer(
            -0.08, 0.08, )
        tf.get_variable_scope().set_initializer(init_op)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        utils.print_out(" created model with fresh parameters")
        step, epoch = 0, 0
    return step, epoch


def restore_a_model(out_dir, model, sess):
    latest_ckpt = tf.train.latest_checkpoint(out_dir)
    if latest_ckpt:
        try:
            print('Try to load from %s' % latest_ckpt)
            model.saver.restore(sess, latest_ckpt)
        except tf.errors.NotFoundError as e:
            utils.print_out("Can't load checkpoint")
            print_variables_in_ckpt(latest_ckpt)
            utils.print_out("%s" % str(e))
            raise e

        sess.run(tf.tables_initializer())
        utils.print_out(
            "  loaded model parameters from %s" % (latest_ckpt))

        step, epoch = sess.run([model.global_step, model.epoch_step])
    else:
       raise Exception()
    return step, epoch


# For Seq2Seq Model
def softmax_cross_entropy_loss(
        logits, labels):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return crossent

def compute_loss(logits, target_output, target_sequence_length,unk_helper=False):
    """Compute optimization loss."""
    max_time = target_output.shape[1].value or tf.shape(target_output)[1]
    crossent = softmax_cross_entropy_loss(
        logits, target_output)
    target_weights = tf.sequence_mask(target_sequence_length, max_time, dtype=tf.float32)
    if unk_helper:
        is_unk = tf.equal(target_output, 0)
        unk_val = tf.cast(is_unk, tf.float32)
        # 减少不必要的unk val被学习到
        unk_val = unk_val / tf.reduce_sum(unk_val, keep_dims=True)
        unk_weights = unk_val * target_weights
        target_weights = tf.where(is_unk, unk_weights, target_weights)
    loss = crossent * target_weights

    return loss


def safe_distribution_log(y):
        return tf.log(tf.clip_by_value(y, 1e-9, 1.0-1e-9))

def gumbel_softmax(probs, temprature=0.1):
    # 输入就必须先是概率
    # probs = tf.nn.softmax(logits)
    log_probs = safe_distribution_log(probs)
    gumbel_rand = tf.random_uniform(tf.shape(log_probs), minval=0.0, maxval=1.0)
    gumbel_rand = -safe_distribution_log(gumbel_rand)
    gumbel_rand = -safe_distribution_log(gumbel_rand)

    gumbel_logits = log_probs + gumbel_rand
    gumbel_logits = gumbel_logits / temprature

    gumbel_probs = tf.nn.softmax(gumbel_logits)

    return gumbel_probs

def sample_from_distribution(P, n_times):
    # [N, batch]
    sample = tf.distributions.Categorical(probs=P).sample(n_times)
    sample_index = tf.transpose(sample)
    return sample_index
