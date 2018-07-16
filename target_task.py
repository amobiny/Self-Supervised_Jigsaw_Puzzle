import tensorflow as tf

imported_graph = tf.train.import_meta_graph('./Results/model_dir/run01/model-6.meta')
print()

checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
model_path = checkpoint_path + '-' + str(epoch)






