# assignment-1-spring2022
Applied ML Spring 2022 HW_1
Eshan Kumar, ek3227

This project consists of Three parts. 
# Part 1
In the first part of the project, I work on an Automobile MPG dataset, which contains characteristics for 8000+ cars. I use linear regression to predict the selling price of these cars. 

### Data Preprocessing
First, I eliminate data points with missing values (>5% of the data). I remove the units from some features and convert them to the same unit scale (Eg. Some mpg is in km/kg, some is km/l, I convert all to km/l and remove the unit strings). 

### Data Visualization
I then plot the relationships between the label (selling price) and the continuous variables on scatterplots, and the relationships between the selling price and the categorical variables in boxplots. This way I can note any positive correlations. 

### Data Preprocessing
Next, I one-hot encode categorical data, and split the data into 60% training data, 20% validation data, and 20% test data. I scale the data according to the mean and SD of the training data, and add a column of 1s to the feature matrices to account for a coefficient for the bias term. 

### Linear Regression Implementation
Rather than using Machine Learning, I find the weight vector using the closed form solution. I built a LinearRegression() class and implement train() to find these weights using the solution, and implement predict() to make predictions on new data using the weights. 

### Initial Training
I then train the Linear Regression Model on the training data with alpha = 0 (regularization parameter: hyperparameter controlling model complexity). Larger alpha gives us smaller weights and decreased model complexity. With alpha = 0, we have vanilla linear regression. We compare the predictions of this trained model to the actual data. 

I then establish a baseline by taking the average of all car selling prices in the test, validation, and development set. This is a good way to tell if our model is doing well. 

### Feature Evaluation
I use the weights to evaluate which features seem to be the biggest factors in determining the selling price of a car. 

### Hyperparameter Tuning (alpha)
I perform a grid search, training multiple models and determining their scores, in order to see which model has elicited the best score, and thus the best parameters to use in the model for production. 

# Part 2
In the second and third part of the project, I work on the voice dataset, which contains the characteristics of 3000+ voice samples from men and women. I use logistic regression to predict whether a voice sample belongs to a man or a woman.

### Data Visualization/Preprocessing
I then plot the relationships between the label (male/female) and the continuous variables on in boxplots. I then plot a correlation matrix, and note the features that have a correlation higher than 0.9. I remove a feature if it is correlated with another over this threshold to prevent collinearity problems, which can include high variation in solutions and numerical instability. It becomes difficult for the model to evaluate the relationship between an independent and dependent variable when independent variables change in unison. 

Then I change the categorical gender labels into an ordinal encoding (1 for females, 0 for males), and split the data into 60% training data, 20% validation data, and 20% test data. I scale the data according to the mean and SD of the training data, and add a column of 1s to the feature matrices to account for a coefficient for the bias term. 

### Logistic Regression Implementation
This has no closed form solution. Therefore, to find the optimum weights, we must use gradient descent. I define a Logistic Regression Class and several helpful functions to perform gradient descent. Using this model, I am able to plot the loss as a model is trained. 

### Hyperparameter Tuning (alpha, epochs, learning rate)
I create a search space across alphas (model complexity, biases against large weights), epochs (number of gradient descent steps), and learning rates (length of step to take while performing gradient descent), and randomly search this space in order to determine an optimal combination of hyperparameters. I then evaluate the accuracy of some models with different hyperparameters on the Test set. 

### Feature Evaluation
I use the weights to evaluate which features seem to be the biggest factors in determining the whether a voice is male or female. 


# Part 3
Using the same dataset as part 2, I use a Dual SVM to do a similar prediction. 

Using sklearns make_pipeline, we create a pipe that preprocesses data (scales it), and then uses a Support Vector Machine to do Classification with two different kernels, a linear kernel and an RBF kernel. The RBF Kernel gave better accuracy on the test dataset. It is usually recommended to use a linear kernel when the number of features is large, and a non-linear RBF (Gaussian) kernel when the number of samples is large. This is because a linear kernel is never more accurate than a properly tuned Gaussian kernel, it is mainly used because it is faster on datasets with many features.

### Hyperparameter Tuning and K-fold crossvalidation (C)
Finally, using the RBF kernel, we tune the hyperparameter “C” using the Grid Search & k-fold cross validation. We take k=5. We found the best score and C-value, and evaluated the test data with this model. 

The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example. For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points.
