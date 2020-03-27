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

For the sampling process I chose a random document d1 and another random document d2 (!=d1)to compare both for training and testing. 
I made sure to create a balanced test and training sample with ~50% positive and negative instances.
When I had a competely random sampling procedure earlier, the model seemed to favour the label 0 over 1, and during parts of the testing classified all testing samples as 0. 
Since the training data was sampled randomly, the distribution of the labels was overall heavy on negative samples, which is why even with only 0s it still performed with an accuracy of around 80%.

I also chose to run the training for 3 epochs, which is defined in the training function of the model.


Example run: python a3_model.py test.csv --size 1000

With the values from the examples (50 features, samplesize = 1000), I got these results on the first three runs:

accuracy: 0.5544455544455544 precision: 0.5763134406261381 recall: 0.5544455544455544 f1_score: 0.521571434091244

accuracy: 0.5394605394605395 precision: 0.5495191737994183 recall: 0.5394605394605395 f1_score: 0.5166568835651731

accuracy: 0.5194805194805194 precision: 0.5640849762301565 recall: 0.5194805194805194 f1_score: 0.42190734995996054


Part 3: 

- there is an optional argument --hidden that takes an integer to determine the size of the layer. If not used, default = 0, therefore there is no hidden layer.

- there is an optional argument --nonlin to set the activation function. If not used, there is no nonlinear layer. You can use tanh or relu.

Example run: python a3_model.py test.csv --size 1000 --hidden 100 --nonlin tanh


According to https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw:
The number of hidden neurons should be between the size of the input layer and the size of the output layer.
The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
The number of hidden neurons should be less than twice the size of the input layer.

Since the number of features given to the NN is 2*50, so 100 I decided to test it on hidden layer sizes between 1 and 100, the output and input size.

Hidden layer size: 100

no nonlinearity: accuracy: 0.5554445554445554 precision: 0.5641016973656224 recall: 0.5554445554445554 f1_score: 0.5410213688888286
Tanh():          accuracy: 0.5754245754245755 precision: 0.5829398090680957 recall: 0.5754245754245755 f1_score: 0.5662655688489688
ReLU():          accuracy: 0.5244755244755245 precision: 0.6111237599609693 recall: 0.5244755244755245 f1_score: 0.4068273908511657


Hidden layer size: 70

no nonlinearity: accuracy: 0.4935064935064935 precision: 0.4934906783240635 recall: 0.4935064935064935 f1_score: 0.4934862729334837
Tanh():          accuracy: 0.5154845154845155 precision: 0.5232693697799281 recall: 0.5154845154845155 f1_score: 0.4653097517879794
ReLU():          accuracy: 0.5424575424575424 precision: 0.5452945386596116 recall: 0.5424575424575424 f1_score: 0.5361782121048123    


Hidden layer size: 60

no nonlinearity: accuracy: 0.5014985014985015 precision: 0.5011210845786107 recall: 0.5014985014985015 f1_score: 0.48659703272693383
Tanh():          accuracy: 0.5404595404595405 precision: 0.5530815859852496 recall: 0.5404595404595405 f1_score: 0.513312572325475
ReLU():          accuracy: 0.5614385614385614 precision: 0.5713507271948831 recall: 0.5614385614385614 f1_score: 0.5445876226918674


Hidden layer size: 50

no nonlinearity: accuracy: 0.5264735264735265 precision: 0.5291428127249023 recall: 0.5264735264735265 f1_score: 0.5131599207927188
Tanh():          accuracy: 0.5464535464535465 precision: 0.5469355368991067 recall: 0.5464535464535465 f1_score: 0.5456356103908886
ReLU():          accuracy: 0.5494505494505495 precision: 0.5507922419729124 recall: 0.5494505494505495 f1_score: 0.5470098135552275

Hidden layer size: 20

no nonlinearity: accuracy: 0.4995004995004995 precision: 0.5006822616156736 recall: 0.4995004995004995 f1_score: 0.44740224629806463
Tanh():          accuracy: 0.5634365634365635 precision: 0.5638400848804334 recall: 0.5634365634365635 f1_score: 0.5629480896265249
ReLU():          accuracy: 0.5324675324675324 precision: 0.540616027329314  recall: 0.5324675324675324 f1_score: 0.5099821527538768


Overall, the differences between the different sizes of the hidden layer were not very big. The model performed slightly better at hidden size 100 than at the rest.
The worst is 70, which is closest to the recommendation of "2/3 the size of the input layer, plus the size of the output layer".
At the smallest tested value 20, the performance increased again to almost the results of 100.

The overall best settings were size 100 and nonlin Tanh, followed by 20 and Tanh.

Between the different activation functions, the model performed the worst whithout any nonlinearity. 
Tanh() and ReLU() improved the results by about the same, with Tanh() outperforming ReLU fo hidden sizes 100 and 20, and ReLU having better results for the sizes between.

The results are overall quite bad, which could be due to the relativly small training sample and the quality of the data.
