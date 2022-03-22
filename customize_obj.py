from deoxys.experiment.postprocessor import DefaultPostProcessor
from deoxys.customize import custom_loss
from deoxys.model.losses import Loss
import tensorflow as tf
import h5py
import os
import numpy as np
import shutil


@custom_loss
class NegativeLogLikelihood(Loss):
    def __init__(self, reduction='auto', name="negative_log_likelihood"):
        super().__init__(reduction, name)

    def call(self, target, prediction):
        target = tf.cast(target, prediction.dtype)

        negative_pred = 1 - prediction
        entropy = tf.math.log(prediction) * target
        negative_entropy = tf.math.log(negative_pred) * negative_pred

        return entropy + negative_entropy


class EnsemblePostProcessor(DefaultPostProcessor):
    def __init__(self, log_base_path='logs',
                 log_path_list=None,
                 map_meta_data=None, **kwargs):

        self.log_base_path = log_base_path
        self.log_path_list = []
        for path in log_path_list:
            merge_file = path + self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            if os.path.exists(merge_file):
                self.log_path_list.append(merge_file)
            else:
                print('Missing file from', path)

        # check if there are more than 1 to ensemble
        assert len(self.log_path_list) > 1, 'Cannot ensemble with 0 or 1 item'

        if map_meta_data:
            if type(map_meta_data) == str:
                self.map_meta_data = map_meta_data.split(',')
            else:
                self.map_meta_data = map_meta_data
        else:
            self.map_meta_data = ['patient_idx']

        # always run test
        self.run_test = True

    def ensemble_results(self):
        # initialize the folder
        if not os.path.exists(self.log_base_path):
            print('Creating output folder')
            os.makedirs(self.log_base_path)

        output_folder = self.log_base_path + self.TEST_OUTPUT_PATH
        if not os.path.exists(output_folder):
            print('Creating ensemble folder')
            os.makedirs(output_folder)

        output_file = output_folder + self.PREDICT_TEST_NAME
        if not os.path.exists(output_file):
            print('Copying template for output file')
            shutil.copy(self.log_path_list[0], output_folder)

        print('Creating ensemble results...')
        y_preds = []
        for file in self.log_path_list:
            with h5py.File(file, 'r') as hf:
                y_preds.append(hf['predicted'][:])

        with h5py.File(output_file, 'a') as mf:
            mf['predicted'][:] = np.mean(y_preds, axis=0)
        print('Ensembled results saved to file')

        return self

    def concat_results(self):
        # initialize the folder
        if not os.path.exists(self.log_base_path):
            print('Creating output folder')
            os.makedirs(self.log_base_path)

        output_folder = self.log_base_path + self.TEST_OUTPUT_PATH
        if not os.path.exists(output_folder):
            print('Creating ensemble folder')
            os.makedirs(output_folder)

        # first check the template
        with h5py.File(self.log_path_list[0], 'r') as f:
            ds_names = list(f.keys())
        ds = {name: [] for name in ds_names}

        # get the data
        for file in self.log_path_list:
            with h5py.File(file, 'r') as hf:
                for key in ds:
                    ds[key].append(hf[key][:])

        # now merge them
        print('creating merged file')
        output_file = output_folder + self.PREDICT_TEST_NAME
        with h5py.File(output_file, 'w') as mf:
            for key, val in ds.items():
                mf.create_dataset(key, data=np.concatenate(val, axis=0))
