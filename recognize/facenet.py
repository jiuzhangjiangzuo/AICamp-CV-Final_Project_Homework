import tensorflow as tf
import numpy as np
import os
import re
import cv2

class FaceNet(object):
    def __init__(self, model_path):
        #create a graph
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            print('Model directory: %s' % model_path)
            meta_file, ckpt_file = self.get_model_filenames(model_path)
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
          
            saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file), input_map=None)
            saver.restore(self.sess, os.path.join(model_path, ckpt_file))


    def get_model_filenames(self, model_dir):
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files)==0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files)>1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file

        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups())>=2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file


    def predict(self, image):
        images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
        embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")

        image = cv2.resize(image, (160, 160))
        # Run forward pass to calculate embeddings
        feed_dict = { images_placeholder: np.stack([image]), phase_train_placeholder:False }
        emb = self.sess.run(embeddings, feed_dict=feed_dict)
        
        return emb[0, :]