# CNN-for-Number-Sorting

# Aim: To design a convnet that sorts numbers. Operators are ReLU, Conv, and Pooling.
# E.g. input: 5, 3, 6, 2; output: 2, 3, 5, 6

There are 2 scripts present in this folder.

1) train_script

This scripts defines the actual convent model and trains it using subsets of 4 numbers
ranging from 0-50.

Various parameters like number of epochs, range of numbers to be trained for, etc.
can be changed and the trained model can be tested on one example given at the end of
the script.

Training and validation(test) loss and accuracy are plotted to observe the performance
of the convnet.

The best model obtained can be saved by uncommenting the last line.

2) test_script

This script loads the best obtained model and sorts some input sequences.

# Observations during various Experiments conducted

1) First the network was trained on 100000 different permutations of numbers from 1 to 50
for about 30 epochs.

However, the results were erroneous.

For [5 3 6 2] , the sequence obtained was [2 2 5 6]

2) Hence, I decided to train the network for about 20 epochs on different permutations
of digits from 0-9 only. 

The loss and accuracy graphs were ideal for training and test.

The result on the test case [5 3 6 2] was also correct => [ 2 3 5 6]
The network is easily able to sort among numbers with less difference such as
[4 3 2 1] was output as [1 2 3 4]

This network is observed to perform poorly on 2 digit numbers since it has been trained on 
only single digit numbers.

Inference: The model may be overfitting since it has seen many repeated cases of the above 
combinations.

3) Lastly I trained the network for 30 epochs for 500000 permutations from 0-49 range to help
the network learn a wider range. It took about 20 minutes to train the model.

This model performs pretty well with atleast 74% accuracy.

The network gives accurate results on numbers it hasn’t seen.

For e.g., it correctly sorts [88,70,93,63] to [63, 70, 88, 93]








