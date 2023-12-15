import pandas as pd
import numpy as np
from pywt import wavedec
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf 
import os 
import random
from datetime import datetime
from typing import List

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
                train_test_split_ratio = 0.8,
                wavelet: str = "haar",
                save_models = True
                ):
        # init variables
        self.experiment_name = experiment_name
        self.sampling_periods = sampling_periods
        self.sampling_statistics = sampling_statistics
        self.sequence_length = sequence_length
        self.sequence_length_em = sequence_length_em
        self.n_step_ahaed = n_step_ahaed
        self.train_test_split_ratio = train_test_split_ratio
        self.wavelet = wavelet
        self.save_models = save_models
        # calculate the number of required past data
        self._biggest_period = max(self.sampling_periods)
        self._number_of_cascaded_modules = len(self.sampling_periods)
        self._required_time_steps = self._biggest_period * self.sequence_length
        self._sorted_sampling_periods = np.sort(sampling_periods)[::-1]
        
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
                              statistics_to_calculate: str) -> pd.DataFrame:
        # calculate statistics
        data = pd.Series(data)
        
        if statistics_to_calculate == "mean":
            return np.mean(data, axis=0)
        elif statistics_to_calculate == "std":
            return np.std(data, axis=0)
        elif statistics_to_calculate == "count":
            return data.count()
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
        for stat in sampling_statistics:
            stat_sequences = []
            for past in past_data:
                cascade_vectors = []
                for period in sorted_sampling_periods:
                    groups = self._build_sampling_groups(past, window_length=period)
                    groups_stats = [self._calculate_statistics(group, stat) for group in groups]
                    # trim the vector to the sequence length
                    trimmed_vector = groups_stats[-sequence_length:]
                    # normalize the vector
                    normalized_vector = np.array(trimmed_vector) - np.mean(trimmed_vector)
                    # calculate the wavelet transform
                    wavelet_coeffs = self._calculate_dwt(normalized_vector, wavelet=wavelet, level=None)
                    cascade_vectors.append(wavelet_coeffs)
                stat_sequences.append(cascade_vectors)
            sequences.append(stat_sequences)
        return *sequences, past_data

    ##############################################################################
    ############################## NEURAL NETWORK FUNCTIONS ######################
    ##############################################################################
    
    # fit a NN with 3 inputs and 1 output, 4 hidden layers 32, 32, 32, 16 relus and dropouts 0.1, 0.1, 0.1, 0.1
    # mse loss, adam optimizer, 300 epochs, batch size 250
    # define the model
    def _train_NN_cm(self, X_train, y_train, sequence_length):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(sequence_length-1,)),
            tf.keras.layers.Dense(sequence_length*2, activation='relu'),
            tf.keras.layers.Dense(sequence_length*4, activation='relu'),
            tf.keras.layers.Dense(sequence_length*4, activation='relu'),
            tf.keras.layers.Dense(sequence_length*2, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        # compile the model
        model.compile(loss='mse', optimizer='adam')
        # fit the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, shuffle=False)
        return model


    # define the EM model
    def _train_NN_em(self, X_train, y_train, sequence_length_em):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(sequence_length_em,)),
            tf.keras.layers.Dense(sequence_length_em*2, activation='relu'),
            tf.keras.layers.Dense(sequence_length_em*4, activation='relu'),
            tf.keras.layers.Dense(sequence_length_em*4, activation='relu'),
            tf.keras.layers.Dense(sequence_length_em*2, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        # compile the model
        model.compile(loss='mse', optimizer='adam')
        # fit the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, shuffle=False)
        return model


    # define the model
    def _train_NN_final(self, X_train, y_train, sequence_length_final):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(sequence_length_final,)),
            tf.keras.layers.Dense(sequence_length_final*2, activation='relu'),
            tf.keras.layers.Dense(sequence_length_final*4, activation='relu'),
            tf.keras.layers.Dense(sequence_length_final*4, activation='relu'),
            tf.keras.layers.Dense(sequence_length_final*2, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        # compile the model
        model.compile(loss='mse', optimizer='adam')
        # fit the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, shuffle=False)
        return model


    # calculate rmse and r2
    def _calculate_scores(self,y_pred, y_test):
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
                   data: np.array) -> None:
        # build sequences
        sequences_mean, past = self._build_sequences(data, 
                                                    self._sorted_sampling_periods, 
                                                    self.sampling_statistics, 
                                                    self.sequence_length, 
                                                    self.wavelet, 
                                                    self._required_time_steps)
        # define X, y
        y = np.asarray(data[self._required_time_steps + self.n_step_ahaed-1:], dtype=np.float32)
        X = np.asarray(sequences_mean[:len(y)], dtype=np.float32)
        
        # train test split
        train_size = int(len(X) * self.train_test_split_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test
    
    def fit(self,
            X_train: np.array, 
            y_train: np.array,
            ):
        ################################ train cascaded modules ########################
        preds = []
        errors = []
        models_cm = []
        # fetch X_train, y_train, X_test, y_test for each cascade module and train the inner NN and calculate train and eval times
        for i in range(self._number_of_cascaded_modules):
            # Fetch related period data
            X_train_cm, y_train_cm = X_train[:,i,:], y_train
            
            # for following CMs, we need to predict the previous errors
            if i>0:
                y_train_cm = errors[i-1]
                
            # Train the inner NN
            model_cm = self._train_NN_cm(X_train_cm, y_train_cm, self.sequence_length)
            
            # calculate the preds and errors
            train_preds_cm = model_cm.predict(X_train_cm)
            preds.append(train_preds_cm)
            train_errors_cm = y_train_cm - train_preds_cm.squeeze()
            errors.append(train_errors_cm)
            
            # save the model
            if self.save_models:
                model_cm.save(f'outputs/models/{self.experiment_name}_CM_{i}.h5')
            models_cm.append(model_cm)
            
        # save the models
        self._models_cm = models_cm
            
        ################################ train error module ########################
        # caluclate the compansated preds and errors
        compansated_preds = np.sum(preds, axis=0)
        # shift time -1
        compansated_preds = compansated_preds[:-1]
        y_train_past = y_train[-1:]
        # calculate the preds and errors
        compansated_errors = y_train_past - compansated_preds.squeeze()
        
        # build train data for error module by windowing compansated errors
        X_train_em = self._build_past_for_each_time_step(compansated_errors, window_length=self.sequence_length_em) 
        X_train_em = np.array(X_train_em[:-1])
        y_train_em = compansated_errors[self.sequence_length_em:]
        
        # fit the error module
        model_em = self._train_NN_em(X_train_em, y_train_em, self.sequence_length_em)
        preds_em = model_em.predict(X_train_em)
        preds.append(preds_em)

        # save the model
        if self.save_models:
            model_em.save(f'outputs/models/{self.experiment_name}_EM.h5')
        self._model_em = model_em
        
        ################################ train final module ########################
        # equalize the lengths of the preds by removing the first elements if size is bigger than the min size else keep the array as it is
        min_size_preds = min([len(pred_arr) for pred_arr in preds])
        preds = [pred_arr[-min_size_preds:] if len(pred_arr) > min_size_preds else pred_arr for pred_arr in preds]
        
        # build the final train data
        X_train_final = np.concatenate(preds, axis=1)
        y_train_final = y_train[-min_size_preds:]
        
        # fit the final model
        model_final = self._train_NN_final(X_train_final, y_train_final, X_train_final.shape[1])
        preds_final_train = model_final.predict(X_train_final)
        self._preds_final_test = preds_final_train
        
        # save the model
        if self.save_models:
            model_final.save(f'outputs/models/{self.experiment_name}_FINAL.h5')
        self._model_final = model_final
        
        ################################ evaluation on train set ########################
        # calculate the final training scores
        self._rmse_train_final, self._r2_train_final, self._mape_train_final = self._calculate_scores(preds_final_train, y_train_final)
        print(f"Final train scores: RMSE: {self._rmse_train_final}, R2: {self._r2_train_final}, MAPE: {self._mape_train_final}")
        return self._rmse_train_final, self._r2_train_final, self._mape_train_final


    def predict(self,
                X_test: np.array,
                y_test_past: np.array):
        # load all models and predict the test data
        preds = []
        for i in range(self._number_of_cascaded_modules):
            # Fetch related period data
            X_test_cm = X_test[:,i,:]
            # Train the inner NN
            model_cm = self._models_cm[i]
            # calculate the preds and errors
            test_preds_cm = model_cm.predict(X_test_cm)
            preds.append(test_preds_cm)
            
        # caluclate the compansated preds and errors
        compansated_preds = np.sum(preds, axis=0)
        # shift time -1
        compansated_preds = compansated_preds[:-1]
        # calculate the preds and errors
        compansated_errors = y_test_past - compansated_preds.squeeze()
        
        # build train data for error module by windowing compansated errors
        X_test_em = self._build_past_for_each_time_step(compansated_errors, window_length=self.sequence_length_em)
        X_test_em = np.array(X_test_em[:-1])

        # load the error module and predict the errors
        model_em = self._model_em
        preds_em = model_em.predict(X_test_em)
        preds.append(preds_em)

        # equalize the lengths of the preds and errors by removing the first elements if size is bigger than the min size else keep the array as it is
        min_size_preds = min([len(pred_arr) for pred_arr in preds])
        preds = [pred_arr[-min_size_preds:] if len(pred_arr) > min_size_preds else pred_arr for pred_arr in preds]

        # build the final train data
        X_test_final = np.concatenate(preds, axis=1)
        
        # load the final model and predict the test data
        model_final = self._model_final
        preds_final_test = model_final.predict(X_test_final)
        self._preds_final_test = preds_final_test

        return preds_final_test