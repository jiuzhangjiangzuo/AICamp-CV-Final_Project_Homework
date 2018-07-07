import tensorflow as tf
import numpy as np
import os
import re
import cv2

class FaceNet(object):
    def __init__(self, model_path):
        # Read model files and init the tf graph and model
        # !!!!!!!!!!!!!!!!!! Implement here !!!!!!!!!!!!!!!
        pass


    def get_model_filenames(self, model_dir):
        """ Returns the path of the meta file and the path of the checkpoint file.
        
        Parameters:
        ----------
        model_dir: string
            the path to model dir.

        Returns:
        -------
        meta_file: string
            the path of the meta file
        ckpt_file: string
            the path of the checkpoint file
        """
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
        """Get the embedding vector of face by facenet

        Parameters:
        ----------
        image: numpy array
            input image array

        Returns:
        -------
        embedding: numpy array
            the embedding vector of face
        """
        
        # !!!!!!!!!!!!!!!!!! Implement here !!!!!!!!!!!!!!!
        return None