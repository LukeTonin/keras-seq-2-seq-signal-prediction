# This file contains code modified licensed under the MIT License:
# Copyright (c) 2017 Guillaume Chevalier # For more information, visit:
# https://github.com/guillaume-chevalier/seq2seq-signal-prediction
# https://github.com/guillaume-chevalier/seq2seq-signal-prediction/blob/master/LICENSE

"""Contains functions to generate artificial data for predictions as well as 
a function to plot predictions."""

import numpy as np
from matplotlib import pyplot as plt

def random_sine(batch_size, steps_per_epoch,
                input_sequence_length, target_sequence_length,
                min_frequency=0.1, max_frequency=10,
                min_amplitude=0.1, max_amplitude=1,
                min_offset=-0.5, max_offset=0.5,
                num_signals=3, seed=43):
    """Produce a batch of signals.

    The signals are the sum of randomly generated sine waves.

    Arguments
    ---------
    batch_size: Number of signals to produce.
    steps_per_epoch: Number of batches of size batch_size produced by the
        generator.
    input_sequence_length: Length of the input signals to produce.
    target_sequence_length: Length of the target signals to produce.
    min_frequency: Minimum frequency of the base signals that are summed.
    max_frequency: Maximum frequency of the base signals that are summed.
    min_amplitude: Minimum amplitude of the base signals that are summed.
    max_amplitude: Maximum amplitude of the base signals that are summed.
    min_offset: Minimum offset of the base signals that are summed.
    max_offset: Maximum offset of the base signals that are summed.
    num_signals: Number of signals that are summed together.
    seed: The seed used for generating random numbers
    
    Returns
    -------
    signals: 2D array of shape (batch_size, sequence_length)
    """
    num_points = input_sequence_length + target_sequence_length
    x = np.arange(num_points) * 2*np.pi/30

    while True:
        # Reset seed to obtain same sequences from epoch to epoch
        np.random.seed(seed)

        for _ in range(steps_per_epoch):
            signals = np.zeros((batch_size, num_points))
            for _ in range(num_signals):
                # Generate random amplitude, frequence, offset, phase 
                amplitude = (np.random.rand(batch_size, 1) * 
                            (max_amplitude - min_amplitude) +
                             min_amplitude)
                frequency = (np.random.rand(batch_size, 1) * 
                            (max_frequency - min_frequency) + 
                             min_frequency)
                offset = (np.random.rand(batch_size, 1) * 
                         (max_offset - min_offset) + 
                          min_offset)
                phase = np.random.rand(batch_size, 1) * 2 * np.pi 
                         

                signals += amplitude * np.sin(frequency * x + phase)
            signals = np.expand_dims(signals, axis=2)
            
            encoder_input = signals[:, :input_sequence_length, :]
            decoder_output = signals[:, input_sequence_length:, :]
            
            # The output of the generator must be ([encoder_input, decoder_input], [decoder_output])
            decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], 1))
            yield ([encoder_input, decoder_input], decoder_output)

def plot_prediction(x, y_true, y_pred):
    """Plots the predictions.
    
    Arguments
    ---------
    x: Input sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_true: True output sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_pred: Predicted output sequence (input_sequence_length,
        dimension_of_signal)
    """

    plt.figure(figsize=(12, 3))

    output_dim = x.shape[-1]
    for j in range(output_dim):
        past = x[:, j] 
        true = y_true[:, j]
        pred = y_pred[:, j]

        label1 = "Seen (past) values" if j==0 else "_nolegend_"
        label2 = "True future values" if j==0 else "_nolegend_"
        label3 = "Predictions" if j==0 else "_nolegend_"

        plt.plot(range(len(past)), past, "o--b",
                 label=label1)
        plt.plot(range(len(past),
                 len(true)+len(past)), true, "x--b", label=label2)
        plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y",
                 label=label3)
    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()

if __name__ == '__main__':

    # This is an example of the plot function and the signal generator
    from matplotlib import pyplot as plt
    gen = random_sine(3, 3, 15, 15)
    for i, data in enumerate(gen):
        input_seq, output_seq = data
        for j in range(input_seq.shape[0]):
            plot_prediction(input_seq[j, :, :],
                            output_seq[j, :, :],
                            output_seq[j, :, :])
        if i > 2:
            break
