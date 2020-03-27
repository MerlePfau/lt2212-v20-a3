# LT2212 V20 Assignment 3

Put any documentation here including any answers to the questions in the 
assignment on Canvas.


Part 1:

- The code assumes an input folder that contains folders with documents (like enron_sample/authors/documents). Therefore if you run the code in the assignment directory you use the date (enron_sample) as the input folder argument. Otherwise the path to your input folder (e.g. home/Merle/StatisticalMethods/A3/enron_sample).

- As the output file you write the name and extension of your output file (e.g. test.csv) if you run it from the assignment directory.

- The number of dimensions is an integer.

- You specify the size of the training data by adding the argument --test with an integer number between 1 and 100. (default is 20)

Example run: python a3_features.py enron_sample test.csv 50 --test 20

For my features I decided to lowercase and check "isalpha()", to clean out any non-words and punctuation.
For the dimesionality reduction I used TruncatedSVD.



Part 2:

- Add the filename of the file you created in part 1 as an argument.

- There is an optional argument --size to determine the size of the training and testing sample batch. 

For the sampling process I chose a random document d1 and another random document d2 (!=d1)to compare both for training and testing. This results in a completely random sample that should be representative the data.

I also chose to run the training for 3 epochs, which is defined in the training function of the model.


Example run: python a3_model.py test.csv --size 10000

With the values from the examples, I got these results on the first three runs:

accuracy: 0.8064 precision: 0.740785727122339 recall: 0.8064 f1_score: 0.759421788931789

accuracy: 0.5842 precision: 0.7275819825672893 recall: 0.5842 f1_score: 0.633369287661967

accuracy: 0.8026 precision: 0.7439299809898795 recall: 0.8026 f1_score: 0.7653767744045419


Part 3: 

- there is an optional argument --hidden that takes an integer to determine the size of the layer. If not used, default = 0, therefore there is no hidden layer.

- there is an optional argument --nonlin to set the activation function. If not used, there is no nonlinear layer. You can use tanh or relu.

Example run: python a3_model.py test.csv --size 10000 --hidden 100 --nonlin tanh


According to https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw:
The number of hidden neurons should be between the size of the input layer and the size of the output layer.
The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
The number of hidden neurons should be less than twice the size of the input layer.

Since the number of features given to the NN is 2*50, so 100 I decided to test it on hidden layer sizes between 1 and 100, the output and input size.

Hidden layer size: 100

no nonlinearity: accuracy: 0.8123 precision: 0.7250776866398734 recall: 0.8123 f1_score: 0.7573142486345905
Tanh():          accuracy: 0.8373 precision: 0.8379419980799674 recall: 0.8373 f1_score: 0.7690212477853432
ReLU():          accuracy: 0.8334 precision: 0.8056928593095436 recall: 0.8334 f1_score: 0.7748548119084394


Hidden layer size: 70

no nonlinearity: accuracy: 0.8249 precision: 0.7375652726902726 recall: 0.8249 f1_score: 0.7609413908365218
Tanh():          accuracy: 0.8347 precision: 0.850703369715796  recall: 0.8347 f1_score: 0.7709296842751653
ReLU():          accuracy: 0.8366 precision: 0.8484376306046896 recall: 0.8366 f1_score: 0.7673466268500948    


Hidden layer size: 60

no nonlinearity: accuracy: 0.7841 precision: 0.7290967835063027 recall: 0.7841 f1_score: 0.7515635424175974
Tanh():          accuracy: 0.834  precision: 0.8430948008132806 recall: 0.834  f1_score: 0.7636022622221977
ReLU():          accuracy: 0.8261 precision: 0.7177110545438425 recall: 0.8261 f1_score: 0.7528051273831365


Hidden layer size: 50

no nonlinearity: accuracy: 0.8208 precision: 0.7175989537223341 recall: 0.8208 f1_score: 0.7454818517425387
Tanh():          accuracy: 0.832  precision: 0.692224           recall: 0.832  f1_score: 0.755703056768559     (only predicted 0s)
ReLU():          accuracy: 0.8363 precision: 0.8449060287592167 recall: 0.8363 f1_score: 0.7626353829685977


Hidden layer size: 20

no nonlinearity: accuracy: 0.7621 precision: 0.7325533988743431 recall: 0.7621 f1_score: 0.7457393186181555
Tanh():          accuracy: 0.8292 precision: 0.7733272661678666 recall: 0.8292 f1_score: 0.7774647786847196
ReLU():          accuracy: 0.8292 precision: 0.8519949445267464 recall: 0.8292 f1_score: 0.7568014139272686


