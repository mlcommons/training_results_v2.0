import h5py
import numpy as np
import tensorflow as tf
import time
import argparse


tf.compat.v1.disable_eager_execution()


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--tf_input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain tfrecord files for the task.")

    parser.add_argument("--hdf5_output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output data dir where .hdf5 files wii be stored.")
    args = parser.parse_args()
    return args


def _parse_record_train_dat(example_proto):
    features = {
        'input_ids':
            tf.io.FixedLenFeature([seq_length], tf.int64),
        'input_mask':
            tf.io.FixedLenFeature([seq_length], tf.int64),
        'segment_ids':
            tf.io.FixedLenFeature([seq_length], tf.int64),
        'masked_lm_positions':
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        'masked_lm_ids':
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        'masked_lm_weights':
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        'next_sentence_labels':
            tf.io.FixedLenFeature([1], tf.int64),
    }
    parsed_feats = tf.io.parse_single_example(example_proto, features=features)
    return parsed_feats


def compareh5andtfrecord(h5, tfrecord, length, count):
    if length == 1:
        if abs(h5 - tfrecord[0]) > 0:
            print("No." + str(count) + " Not Equal")
    else:
        for i in range(length):
            if abs(h5[i] - tfrecord[i]) > 0:
                print("No." + str(count) + " Not Equal")
                break


# transform tfrecord to hdf5
def get_sample_num(input):
    dataset = tf.data.TFRecordDataset(input)
    dataset = dataset.map(_parse_record_train_dat)
    iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_sample = iter.get_next()
    num = 0
    with tf.compat.v1.Session() as sess:
        while True:
            try:
                sess.run(next_sample)
                num += 1
            except BaseException as e:
                break
    return num


# transform tfrecord to hdf5
def write_tfrecord_to_hdf5(input, hdf5path, num_eval_samples):
    dataset = tf.data.TFRecordDataset(input)
    dataset = dataset.map(_parse_record_train_dat)
    iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_sample = iter.get_next()

    with h5py.File(hdf5path, 'w') as f:
        f.create_dataset('input_ids', shape=(num_eval_samples, seq_length), dtype="int32")
        f.create_dataset('input_mask', shape=(num_eval_samples, seq_length), dtype="int8")
        f.create_dataset('segment_ids', shape=(num_eval_samples, seq_length), dtype="int8")
        f.create_dataset('masked_lm_positions', shape=(num_eval_samples, max_predictions_per_seq), dtype="int32")
        f.create_dataset('masked_lm_ids', shape=(num_eval_samples, max_predictions_per_seq), dtype="int32")
        f.create_dataset('next_sentence_labels', shape=(num_eval_samples,), dtype="int8")
        with tf.compat.v1.Session() as sess:
            for i in range(num_eval_samples):
                example = sess.run(next_sample)
                input_ids = example['input_ids']
                input_mask = example['input_mask']
                segment_ids = example['segment_ids']
                masked_lm_positions = example['masked_lm_positions']
                masked_lm_ids = example['masked_lm_ids']
                next_sentence_labels = example['next_sentence_labels']

                f['input_ids'][i] = input_ids
                f['input_mask'][i] = input_mask
                f['segment_ids'][i] = segment_ids
                f['masked_lm_positions'][i] = masked_lm_positions
                f['masked_lm_ids'][i] = masked_lm_ids
                f['next_sentence_labels'][i] = next_sentence_labels
                if i % 1000 == 0 or (i+1) == num_eval_samples:
                    print("Finish " + str(i+1) + " samples")
        f.close()


if __name__ == '__main__':
    args = parse_arguments()
    start = time.time()
    seq_length = 512
    max_predictions_per_seq = 76
    num_eval_samples = get_sample_num(args.tf_input_dir)
    print(args.tf_input_dir + " have " + str(num_eval_samples) + " samples in total.")
    write_tfrecord_to_hdf5(args.tf_input_dir, args.hdf5_output_dir, num_eval_samples)
    end = time.time()
    print("tranform "+args.tf_input_dir+" costs: "+str((end-start)/60)+" min.")

