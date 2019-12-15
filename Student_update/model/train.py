#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pickle

import utils.data_helpers as data_helpers
import model.MLP as MLP

def load_preprocessing():
    with open("Pickle/parameters.bin", "rb") as f:
        parameters = pickle.load(f)

    with open("Pickle/data_info.bin", "rb") as f:
        data_info = pickle.load(f)

    return parameters, data_info

def create_model(session, parameters, data_info):

    Model = MLP.MLP(parameters=parameters, data_info=data_info)
    session.run(tf.global_variables_initializer())

    return Model

def train():
    print("  >> Loading preprocessing information...", "\n")
    parameters, data_info = load_preprocessing()

    print ("  >> Loading Train Data...", "\n")
    train_data = data_info.train_data
    valid_data = data_info.valid_data

    valid_loss_history = []
    valid_acc_history = []
    train_loss_history = []
    train_acc_history = []

    bad_counter = 0
    previous_min_valid_loss =1000

    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    with tf.Session(config = session_conf) as sess:
        Model = create_model(sess,parameters, data_info)

        if Model.global_epoch_step.eval() + 1 > parameters['n_epoch']:
            print ("  >> Current Epoch: {}, Max Epoch: {}".format(Model.global_epoch_step.eval(), parameters['n_epoch']))
            print ("  >> End of Training....")
            #exit(-1)
            return

        for epoch_idx in range(parameters['n_epoch']):
            try:
                batches = data_helpers.batch_iter(parameters, train_data)
                for minibatch in batches:
                    #print (minibatch)
                    input_indices, target_indices = data_helpers.get_minibatch(\
                            dataset = train_data,\
                            minibatch_seq = minibatch)

                    feed_dict = {
                        Model.X     :   input_indices,\
                        Model.Y     :   target_indices}

                    #   Training model......
                    _, global_step, minibatch_loss, minibatch_accuracy = sess.run(\
                        [Model.train_op, Model.global_step, Model.loss, Model.accuracy], feed_dict)


                    #   Validation Check
                    if (global_step+1) % parameters['evaluation_every'] == 0 :

                        #Validation Set
                        valid_pred, valid_loss, valid_accuracy = valid_check(current_session = sess,\
                                                                            valid_data = valid_data,\
                                                                            Model = Model)

                        # 매 "evaluation_every" step마다 train의 결과를 저장!!!
                        train_loss_history.append(minibatch_loss)
                        train_acc_history.append(minibatch_accuracy)
                        valid_loss_history.append(valid_loss)
                        valid_acc_history.append(valid_accuracy)

                        print ("")
                        print ("  >> Global_Step # {} at {}-epoch".format(global_step, Model.global_epoch_step.eval()))
                        print ("        - Train Loss (Validation_Loss) : {:,.2f} ({:,.2f})".format(minibatch_loss, valid_loss))
                        print ("        - Train Accuracy (Validation Accuracy) : {:,.2f} ({:.2f})".format(minibatch_accuracy, valid_accuracy))
                        print ("")

                        if valid_loss <= previous_min_valid_loss:
                            bad_counter = 0
                            previous_min_valid_loss = valid_loss

                            #save the model checkpoint
                            checkpoint_path = os.path.join(os.path.join(parameters['save_dir'], parameters['model_name']), 'ckpt')
                            saver = tf.train.Saver()
                            saver.save(sess, checkpoint_path, global_step)
                            print("  >> Saving the current model with loss {:,.2f} at {}".format(valid_loss, checkpoint_path))
                            print("")

                        else:
                            bad_counter += 1

                        # Early Stopping
                        if bad_counter > parameters['patience']:
                            print ("  >> EARLY STOPPING with bad_counter {}".format(bad_counter))
                            print ("  >> Training Process Terminated....")
                            #exit(-1)
                            return

                Model.global_epoch_step_op.eval()   #Increment Global_epoch_step

            except KeyboardInterrupt:
                print ("  >> Interrupted by user at {}-epoch, {}-global_step".format(Model.global_epoch_step.eval(), global_step))
                # save the model checkpoint
                saver = tf.train.Saver()
                checkpoint_path = os.path.join(parameters['save_dir'], parameters['model_name'])
                saver.save(sess, checkpoint_path, global_step)
                print("     - Saving the model with {}-epoch in {}".format(Model.global_epoch_step.eval(), checkpoint_path))
                print("     - Training Process Terminated....")
                #exit(-1)
                return

        print ("  >> Save the last model...")
        saver = tf.train.Saver()
        checkpoint_path = os.path.join(parameters['save_dir'], 'MLP.ckpt')
        saver.save(sess, checkpoint_path, global_step = global_step)

    print ("  >> End of Training...")
    print ("")
    print ("")

def valid_check(current_session, valid_data, Model):
    valid_input_indices, valid_target_indices = data_helpers.get_minibatch( \
        dataset=valid_data, \
        minibatch_seq=np.arange(len(valid_data)))

    feed_dict = {
        Model.X: valid_input_indices, \
        Model.Y: valid_target_indices
    }

    valid_pred, valid_loss, valid_accuracy = current_session.run([Model.prediction, Model.loss, Model.accuracy], \
                                                      feed_dict=feed_dict)

    return valid_pred, valid_loss, valid_accuracy


if __name__ == '__main__':
    train()