import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def load_pipeline_config(config_file):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_file, "r") as f:
        proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
    return pipeline_config

def update_pipeline_config(pipeline_config, labels, base_model_path, labelmap_file, train_record_file, test_record_file):
    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(base_model_path, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = labelmap_file
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [train_record_file]
    pipeline_config.eval_input_reader[0].label_map_path = labelmap_file
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [test_record_file]

def save_pipeline_config(pipeline_config, config_file):
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(config_file, "wb") as f:
        f.write(config_text)
