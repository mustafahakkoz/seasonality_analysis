import pandas as pd
import numpy as np
from pywt import wavedec, Wavelet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf 
import os 
import random
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt

# This is for experiment consistency. 
def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
reset_random_seeds(42)

class Pecnet():
    def __init__(self,
                experiment_name: str,
                sampling_periods: List[int] = [1, 4, 8],
                sampling_statistics: List[str] = ["mean"],
                sequence_length: int = 4,
                sequence_length_em: int = 4,
                n_step_ahaed: int = 1, # test this one before using it
                test_size = 0.2,
                wavelet: str = "haar",
                save_models = True
                ):
        # init variables
        self.experiment_name = experiment_name
        self.sampling_periods = sampling_periods
        self.sampling_statistics = sampling_statistics
        # check if sequence_length and sequence_length_em is a power of 2
        if np.log2(sequence_length) % 1 != 0:
            raise ValueError("sequence_length must be a power of 2")
        else:
            self.sequence_length = sequence_length
        if np.log2(sequence_length_em) % 1 != 0:
            raise ValueError("sequence_length_em must be a power of 2")
        self.sequence_length_em = sequence_length_em
        self.n_step_ahaed = n_step_ahaed
        self.test_size = test_size
        self.wavelet = wavelet
        self.save_models = save_models
        # calculate the number of required past data
        # self._n_wavelet_coefficients = len(Wavelet(wavelet).dec_lo)
        # self._max_level = int(np.log2(self.sequence_length/self._n_wavelet_coefficients))
        # self_n_coeffs_returned = self._max_level + 2
        self._biggest_period = max(self.sampling_periods)
        self._n_periods = len(self.sampling_periods)
        self._n_statistics = len(self.sampling_statistics)
        self._n_cascaded_modules = self._n_statistics * self._n_periods
        self._required_time_steps = self._biggest_period * self.sequence_length
        self._sorted_sampling_periods = np.sort(sampling_periods)[::-1]
        self._preds = []
        self._errors = []
        self._models_cm = []
        self._test_preds = []
        
    ########################################################################################
    ############################## PREPROCESSING UTILITY FUNCTIONS #########################
    ########################################################################################

    def _build_past_for_each_time_step(self, values, window_length):
        past_data = []

        # for each time step, we need to build the past data
        for i in range(0, len(values)-window_length+1):
            window = values[i:i+window_length]
            past_data.append(window)

        return past_data


    def _build_sampling_groups(self, values, window_length=7):
        sampling_windows = []

        # Add full windows in reverse order
        for i in range(len(values), 0, -window_length):
            window = values[i-window_length:i]
            sampling_windows.append(window)

        # Add the last window with remaining elements
        remainder_size = len(values) % window_length
        if remainder_size > 0:
            last_window = values[:remainder_size]
            sampling_windows[-1] = last_window

        # reverse the order of windows
        sampling_windows = sampling_windows[::-1]
        return sampling_windows


    def _calculate_statistics(self,
                              data: np.ndarray,
                              statistics_to_calculate: str) -> [int, float]:
        # calculate statistics
        data = pd.Series(data)
        
        if statistics_to_calculate == "mean":
            return np.mean(data, axis=0)
        elif statistics_to_calculate == "std":
            return np.std(data, axis=0)
        elif statistics_to_calculate == "max":
            return np.max(data, axis=0)
        elif statistics_to_calculate == "min":
            return np.min(data, axis=0)
        elif statistics_to_calculate == "median":
            return np.median(data, axis=0)
        elif statistics_to_calculate == "mode":
            return data.mode()[0]
        elif statistics_to_calculate == "count":
            return len(data)
        elif statistics_to_calculate == "skew":
            return data.skew()
        elif statistics_to_calculate == "kurtosis":
            return data.kurtosis()
        else:
            raise ValueError(f"Unsupported statistics: {statistics_to_calculate}")
        
        
    def _calculate_dwt(self,
                        window: np.ndarray, 
                        wavelet: str,
                        level: int) -> List[float]:
        coeffs = wavedec(window, wavelet, mode="zero", level=level)
        return np.concatenate(coeffs)[1:]


    def _build_sequences(self, data, sorted_sampling_periods, sampling_statistics, sequence_length, wavelet, required_time_steps):
        # build past data
        past_data = self._build_past_for_each_time_step(data, window_length=required_time_steps)
        # build sequences
        sequences = []
        sequence_means = []
        for past in past_data:
            cascade_sequences = []
            cascade_sequence_means = []
            for period in sorted_sampling_periods:
                stat_sequences = []
                stat_sequence_means = []
                for stat in sampling_statistics:
                    # sub-sampling groups
                    groups = self._build_sampling_groups(past, window_length=period)
                    # calculate statistics for each group
                    groups_stats = [self._calculate_statistics(group, stat) for group in groups]
                    # trim the vector to the sequence length
                    trimmed_vector = groups_stats[-sequence_length:]
                    # normalize the vector
                    vector_mean = np.mean(trimmed_vector)
                    stat_sequence_means.append(vector_mean)
                    normalized_vector = np.array(trimmed_vector) - vector_mean
                    # calculate the wavelet transform
                    wavelet_coeffs = self._calculate_dwt(normalized_vector, wavelet=wavelet, level=None)
                    stat_sequences.append(wavelet_coeffs)
                cascade_sequences.append(stat_sequences)
                cascade_sequence_means.append(stat_sequence_means)
            sequences.append(cascade_sequences)
            sequence_means.append(cascade_sequence_means)
        return np.array(sequences), np.array(sequence_means), np.array(past_data)

    ##############################################################################
    ############################## NEURAL NETWORK FUNCTIONS ######################
    ##############################################################################
    
    # fit a NN with 3 inputs and 1 output, 4 hidden layers 32, 32, 32, 16 relus and dropouts 0.1, 0.1, 0.1, 0.1
    # mse loss, adam optimizer, 300 epochs, batch size 250
    # define the model
    def _train_NN_cm(self, X_train, y_train, sequence_length):
        input_size = sequence_length-1
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(input_size*2, activation='relu'),
            tf.keras.layers.Dense(input_size*4, activation='relu'),
            tf.keras.layers.Dense(input_size*2, activation='relu'),
            tf.keras.layers.Dense(input_size, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        # compile the model
        model.compile(loss='mse', optimizer='adam')
        # fit the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        return model


    # define the EM model
    def _train_NN_em(self, X_train, y_train, sequence_length_em):
        input_size = sequence_length_em-1
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(input_size*2, activation='relu'),
            tf.keras.layers.Dense(input_size*4, activation='relu'),
            tf.keras.layers.Dense(input_size*2, activation='relu'),
            tf.keras.layers.Dense(input_size, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        # compile the model
        model.compile(loss='mse', optimizer='adam')
        # fit the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        return model


    # define the model
    def _train_NN_final(self, X_train, y_train, sequence_length_final):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(sequence_length_final,)),
            tf.keras.layers.Dense(sequence_length_final*2, activation='relu'),
            tf.keras.layers.Dense(sequence_length_final*4, activation='relu'),
            tf.keras.layers.Dense(sequence_length_final*2, activation='relu'),
            tf.keras.layers.Dense(sequence_length_final, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        # compile the model
        model.compile(loss='mse', optimizer='adam')
        # fit the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        return model


    # calculate rmse and r2
    def _calculate_scores(self, y_test, y_pred):
        # calculate rmse
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # calculate r2
        r2 = r2_score(y_test, y_pred)
        # calculate mape
        mape = mean_absolute_percentage_error(y_test, y_pred)
        return rmse, r2, mape

    ####################################################################
    ############################## MAIN FUNCTIONS ######################
    ####################################################################
    def preprocess(self,
                   data: np.array) -> List[np.array]:
        # build sequences
        sequences, sequence_means, past = self._build_sequences(data, 
                                                                self._sorted_sampling_periods, 
                                                                self.sampling_statistics, 
                                                                self.sequence_length, 
                                                                self.wavelet, 
                                                                self._required_time_steps)
        # define X, y
        y = np.asarray(data[self._required_time_steps + self.n_step_ahaed-1:], dtype=np.float32)
        X = np.asarray(sequences[:len(y)], dtype=np.float32)
        
        # define X_means
        X_means = np.asarray(sequence_means[:len(y)], dtype=np.float32)

        # test size may be a ratio or an integer
        if self.test_size < 1:
            test_size = int(len(X) * self.test_size)
        else:
            test_size = self.test_size
            
        # train test split
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        self.X_train_means, self.X_test_means = X_means[:-test_size], X_means[-test_size:]
        
        # normalize y_train and y_test by lowest period's mean
        y_train = y_train - self.X_train_means[:,-1,0]
        y_test = y_test - self.X_test_means[:,-1,0]
        return X_train, X_test, y_train, y_test
    
    def fit(self,
            X_train: np.array, 
            y_train: np.array
            ) -> List[float]:
        ################################ train cascaded modules ########################
        # fetch X_train, y_train, X_test, y_test for each cascade module and train the inner NN and calculate train and eval times
        cascade_ctr = 0
        for p, period in enumerate(self.sampling_periods):
            for s, stat in enumerate(self.sampling_statistics):
                # if the statistics includes std, skip the last period (1-day) since it is a zero vector.
                if stat == "std" and period == 1:
                    continue
                
                # Fetch related period data
                X_train_cm, y_train_cm = X_train[:,p,s,:], y_train
                
                # for following CMs, we need to predict the previous errors
                if cascade_ctr>0:
                    y_train_cm = self._errors[cascade_ctr-1]
                    
                # Train the inner NN
                model_cm = self._train_NN_cm(X_train_cm, y_train_cm, self.sequence_length)
                
                # calculate the preds and errors
                train_preds_cm = model_cm.predict(X_train_cm)
                self._preds.append(train_preds_cm)
                train_errors_cm = y_train_cm - train_preds_cm.squeeze()
                self._errors.append(train_errors_cm)
                
                # iterate the counter
                cascade_ctr += 1
                
                # save the model
                if self.save_models:
                    model_cm.save(f'outputs/models/{self.experiment_name}_CM_{cascade_ctr}.h5')
                self._models_cm.append(model_cm)
        
        # update the number of cascaded modules
        self._n_cascaded_modules = cascade_ctr
            
        ################################ train error module ########################
        # calculate the compansated preds and errors
        compansated_preds = self._preds[0] - np.sum(self._preds[1:], axis=0)
        
        # calculate the error of the compansated preds
        compansated_preds_errors = compansated_preds.squeeze() - y_train
        
        # shift time -1
        compansated_preds_errors = compansated_preds_errors[:-1]
        
        # build train data for error module by windowing compansated_preds_errors
        X_train_em = self._build_past_for_each_time_step(compansated_preds_errors, window_length=self.sequence_length_em) 
        X_train_em = np.array(X_train_em[:-1])
        y_train_em = compansated_preds_errors[self.sequence_length_em:]
        
        # normalize X_train_em and y_train_em by subtracting mean of every row
        X_train_em_means = np.mean(X_train_em, axis=1, keepdims=True)
        X_train_em_normalized = X_train_em - X_train_em_means
        y_train_em_normalized = y_train_em - X_train_em_means.squeeze()
        
        # calculate the wavelet transform
        wavelet_coeffs_em = np.apply_along_axis(self._calculate_dwt, 1, X_train_em_normalized, wavelet=self.wavelet, level=None)

        # fit the error module
        model_em = self._train_NN_em(wavelet_coeffs_em, y_train_em_normalized, self.sequence_length_em)
        preds_em = model_em.predict(wavelet_coeffs_em)
        
        # denormalize the preds
        preds_em_denormalized = preds_em + X_train_em_means
        
        # save the preds
        self._preds.append(preds_em_denormalized)

        # save the model
        if self.save_models:
            model_em.save(f'outputs/models/{self.experiment_name}_EM.h5')
        self._model_em = model_em
        
        ################################ train final module ########################        
        # equalize the lengths of the preds and errors by removing first 5 elements of cascaded module's preds and errors
        self._num_elements_to_remove = self.sequence_length_em + 1 # 1 for time shifting
        self._preds = [pred_arr[self._num_elements_to_remove:] if i < self._n_cascaded_modules else pred_arr for i, pred_arr in enumerate(self._preds)]

        # build the final train data
        X_train_final = np.concatenate(self._preds, axis=1)
        y_train_final = y_train[self._num_elements_to_remove:]
        
        # fit the final model
        model_final = self._train_NN_final(X_train_final, y_train_final, X_train_final.shape[1])
        preds_final_train = model_final.predict(X_train_final)
        
        # denormalize the pred and y_train_final by lowest period's mean
        preds_final_train_denormalized = preds_final_train.squeeze() + self.X_train_means[:,-1,0][self._num_elements_to_remove:]
        y_train_final_denormalized = y_train_final + self.X_train_means[:,-1,0][self._num_elements_to_remove:]
        
        # save the model
        if self.save_models:
            model_final.save(f'outputs/models/{self.experiment_name}_FINAL.h5')
        self._model_final = model_final
        
        ################################ evaluation on train set ########################
        # calculate the final training scores
        self._rmse_train_final, self._r2_train_final, self._mape_train_final = self._calculate_scores(y_train_final_denormalized, preds_final_train_denormalized)
        print(f"Final train scores: RMSE: {self._rmse_train_final}, R2: {self._r2_train_final}, MAPE: {self._mape_train_final}")
        return self._rmse_train_final, self._r2_train_final, self._mape_train_final


    def predict(self,
                X_test: np.array,
                y_test: np.array) -> List[np.array]:
        # load all models and predict the test data
        cascade_ctr = 0
        for p, period in enumerate(self.sampling_periods):
            for s, stat in enumerate(self.sampling_statistics):
                # if the statistics includes std, skip the last period (1-day) since it is a zero vector.
                if stat == "std" and period == 1:
                    continue
                
                # Fetch related period data
                X_test_cm = X_test[:,p,s,:]
                
                # Load the inner NN
                model_cm = self._models_cm[cascade_ctr]
                
                # calculate the preds and errors
                test_preds_cm = model_cm.predict(X_test_cm)
                self._test_preds.append(test_preds_cm)
                
                # iterate the counter
                cascade_ctr += 1
            
        # calculate the compansated preds and errors
        compansated_preds_errors = self._test_preds[0] - np.sum(self._test_preds[1:], axis=0)
        
        # calculate the preds and errors
        compansated_preds_errors = compansated_preds_errors.squeeze() - y_test

        # shift time -1
        compansated_preds_errors = compansated_preds_errors[:-1]

        # build train data for error module by windowing compansated errors
        X_test_em = self._build_past_for_each_time_step(compansated_preds_errors, window_length=self.sequence_length_em)
        X_test_em = np.array(X_test_em[:-1])

        # normalize X_test_em by subtracting mean
        X_test_em_means = np.mean(X_test_em, axis=1, keepdims=True)
        X_test_em_normalized = X_test_em - X_test_em_means

        # calculate the wavelet transform
        wavelet_coeffs_em = np.apply_along_axis(self._calculate_dwt, 1, X_test_em_normalized, wavelet=self.wavelet, level=None)

        # load the error module and predict the errors
        model_em = self._model_em
        preds_em = model_em.predict(wavelet_coeffs_em)
        
        # denormalize the preds
        preds_em = preds_em + X_test_em_means
        
        # save the preds
        self._test_preds.append(preds_em)

        # equalize the lengths of the preds and errors by removing first 5 elements of cascaded module's preds and errors
        self._test_preds = [pred_arr[self._num_elements_to_remove:] if i < self._n_cascaded_modules else pred_arr for i, pred_arr in enumerate(self._test_preds)]

        # build the final train data
        X_test_final = np.concatenate(self._test_preds, axis=1)
        y_test_final = y_test[self._num_elements_to_remove:]
        
        # load the final model and predict the test data
        model_final = self._model_final
        preds_final_test = model_final.predict(X_test_final)
        self._preds_final_test = preds_final_test
        
        # denormalize the pred and y_train_final by lowest period's mean
        preds_final_test_denormalized = preds_final_test.squeeze() + self.X_test_means[:,-1,0][self._num_elements_to_remove:]
        y_test_final_denormalized = y_test_final + self.X_test_means[:,-1,0][self._num_elements_to_remove:]

        return preds_final_test_denormalized, y_test_final_denormalized