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

To caluculate the loss, I chose to use BCELoss(), since it is a binary classification.


Example run: python a3_model.py test.csv --size 10000

With the values from the examples (50 features, samplesize = 10000), I got these results on the first three runs:

accuracy: 0.5562443755624438 precision: 0.5616341093332067 recall: 0.5562443755624438 f1_score: 0.5462326100964033

accuracy: 0.5758424157584242 precision: 0.580107951876013 recall: 0.5758424157584242 f1_score: 0.5701741263322165

accuracy: 0.5241475852414759 precision: 0.531645608049904 recall: 0.5241475852414759 f1_score: 0.4938569345467199


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

no nonlinearity: accuracy: 0.506049395060494 precision: 0.5313719101100688 recall: 0.506049395060494 f1_score: 0.38203517904426276
Tanh():          accuracy: 0.7502249775022498 precision: 0.7502572023508456 recall: 0.7502249775022498 f1_score: 0.7502175654538614
ReLU():          accuracy: 0.7283271672832716 precision: 0.7343890168817426 recall: 0.7283271672832716 f1_score: 0.7265488441610882


Hidden layer size: 80

no nonlinearity: accuracy: 0.5212478752124787 precision: 0.5403333740157472 recall: 0.5212478752124787 f1_score: 0.45659396849809136
Tanh():          accuracy: 0.6765323467653235 precision: 0.7119493707367333 recall: 0.6765323467653235 f1_score: 0.6623948624684826
ReLU():          accuracy: 0.6504349565043496 precision: 0.6586277666015621 recall: 0.6504349565043496 f1_score: 0.6458370617384518   


Hidden layer size: 60

no nonlinearity: accuracy: 0.5032496750324967 precision: 0.5037881514062184 recall: 0.5032496750324967 f1_score: 0.4828232641570284
Tanh():          accuracy: 0.6857314268573143 precision: 0.7015758178406429 recall: 0.6857314268573143 f1_score: 0.6794083767577425
ReLU():          accuracy: 0.5912408759124088 precision: 0.6277789217368286 recall: 0.5912408759124088 f1_score: 0.5598608414455218


Hidden layer size: 40

no nonlinearity: accuracy: 0.49695030496950304 precision: 0.4958721914414103 recall: 0.49695030496950304 f1_score: 0.4590439092051328
Tanh():          accuracy: 0.6432356764323568  precision: 0.6685948286928081 recall: 0.6432356764323568  f1_score: 0.6293393779874773
ReLU():          accuracy: 0.7156284371562843  precision: 0.7195589358992553 recall: 0.7156284371562843  f1_score: 0.7143406019430337

Hidden layer size: 20

no nonlinearity: accuracy: 0.5236476352364764 precision: 0.5311904940519587 recall: 0.5236476352364764 f1_score: 0.4933298929767751
Tanh():          accuracy: 0.6913308669133087 precision: 0.6922934923153071 recall: 0.6913308669133087 f1_score: 0.6909381545425872
ReLU():          accuracy: 0.5768423157684232 precision: 0.630598969153266 recall: 0.5768423157684232 f1_score: 0.5281857050877707


Overall, between the different activation functions, the model performed by far the worst whithout any nonlinearity, consistently over all the sizes.

The model performed the best at hidden size 100 for both Tanh() and ReLU() as the activation functions with f1 scores of 75 and 72%.

For Tanh() the results got worse with smaller hidden layers, while ReLu() performed really well with a hidden layer with 40 neurons (71%).



Part Bonus:

- To plot the precision-recall curve, call the file a3_model_2.py.

- As the output file you write the name and extension of your output file (e.g. plot.png) to the optional argument --out if you run it from the assignment directory.

- You can still use the --nonlin argument from part3.

Example run: python a3_model_2.py test.csv --size 10000 --nonlin tanh --out plot.png

I uploaded the plot from this example to my github repo.

I chose to test the hidden layer sizes 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 and 100, similarly to part 3.
For the plot, I sorted the results after the result. Unfortunately I did not manage to include the respective hidden layer sizes in the graph,
so that the connection between each point on the line and the size in not visible in the graph.