import tensorflow as tf
import pandas as pd
from tensorflow.python.lib.io.tf_record import TFRecordWriter
import csv
from sklearn.model_selection import train_test_split
import json
import time

# Preprocessing

## First convert the corpus 2 into a cvs file
dataset_path = "corpus2_train.labels"
raw_doc = []
raw_cat = []
with open(dataset_path, "r") as file:
    for line in file:
        line = line.strip()
        doc, category = line.split(' ')
        raw_doc.append(doc)
        raw_cat.append(category)
with open('raw_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "label", "sentence"])
    count = 1
    for i in range(len(raw_doc)):
        each_doc = raw_doc[i]
        with open(each_doc, "r") as file:
            writer.writerow([count, raw_cat[i], file.read()])
        count = count + 1

raw_data_path = "raw_dataset.csv"
destination_folder = "./dataset"

train_test_ratio = 0.92
train_valid_ratio = 0.92

# Read raw data
df_raw = pd.read_csv(raw_data_path)

# Prepare columns
df_raw['label'] = (df_raw['label'] == 'O').astype('int')
df_raw = df_raw.reindex(columns=['idx', 'sentence', 'label'])

# Split according to label
df_indoor = df_raw[df_raw['label'] == 0]
df_outdoor = df_raw[df_raw['label'] == 1]

# raw = pd.concat([df_indoor, df_outdoor], ignore_index=True, sort=True)
# raw.head()
# Train-test split
df_indoor_full_train, df_indoor_test = train_test_split(df_indoor, train_size=train_test_ratio, random_state=1)
df_outdoor_full_train, df_outdoor_test = train_test_split(df_outdoor, train_size=train_test_ratio, random_state=1)

# Train-valid split
df_indoor_train, df_indoor_valid = train_test_split(df_indoor_full_train, train_size=train_valid_ratio, random_state=1)
df_outdoor_train, df_outdoor_valid = train_test_split(df_outdoor_full_train, train_size=train_valid_ratio,
                                                      random_state=1)

# Concatenate splits of different labels
df_train = pd.concat([df_indoor_train, df_outdoor_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_indoor_valid, df_outdoor_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_indoor_test, df_outdoor_test], ignore_index=True, sort=False)
df_train.sample(frac=1)
df_valid.sample(frac=1)
df_test.sample(frac=1)

# Using only numpy representation of the values
train_csv = df_train.values
validate_csv = df_valid.values
test_csv = df_test.values

def create_tf_example(features, label):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[features[0]])),
        'sentence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[1].encode('utf-8')])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))

    return tf_example


def convert_csv_to_tfrecord(csv, file_name):
    start_time = time.time()
    writer = TFRecordWriter(file_name)
    for index, row in enumerate(csv):
        try:
            if row is None:
                raise Exception('Row Missing')
            if row[0] is None or row[1] is None or row[2] is None:
                raise Exception('Value Missing')
            if row[1].strip() is '':
                raise Exception('Utterance is empty')
            features, label = row[:-1], row[-1]
            example = create_tf_example(features, label)
            writer.write(example.SerializeToString())
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
    writer.close()
    print(f"{file_name}: --- {(time.time() - start_time)} seconds ---")


def generate_json_info(local_file_name):
    info = {"train_length": len(df_train), "validation_length": len(df_valid),
            "test_length": len(df_test)}

    with open(local_file_name, 'w') as outfile:
        json.dump(info, outfile)


convert_csv_to_tfrecord(train_csv, "data/train.tfrecord")
convert_csv_to_tfrecord(validate_csv, "data/validate.tfrecord")
convert_csv_to_tfrecord(test_csv, "data/test.tfrecord")

generate_json_info("data/corpus2_info.json")

# tr_ds = tf.data.TFRecordDataset("data/train.tfrecord")
# feature_spec = {
#     'index': tf.io.FixedLenFeature([], tf.int64),
#     'text': tf.io.FixedLenFeature([], tf.string),
#     'label': tf.io.FixedLenFeature([], tf.int64)
# }
#
#
# def parse_example(example_proto):
#     # Parse the input tf.Example proto using the dictionary above.
#     return tf.io.parse_single_example(example_proto, feature_spec)
#
#
# tr_parse_ds = tr_ds.map(parse_example)
# dataset_iterator = iter(tr_parse_ds)
# print(dataset_iterator.get_next())