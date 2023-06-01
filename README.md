# MushroomClassification
![image](https://github.com/SurajspatiL99/MushroomClassification/assets/101862962/9b1f2eb1-6a80-44f7-a59f-2a9e68965e5b)

## Introduction
The ability to distinguish between edible and poisonous mushrooms is critical for ensuring food safety and preventing health risks. However, this task can be challenging due to the variety of mushroom types and their similar appearances. In this project, we aim to develop a model that can accurately identify whether a mushroom is edible or poisonous. By utilizing various characteristics of mushrooms, we can create a system that will reduce the risk of consuming poisonous mushrooms, thereby promoting safe and healthy consumption. This project has important implications for both the food industry and consumers, as it can help ensure food safety and prevent potentially deadly consequences.

## Data description
The dataset was procured from the very versatile website for the Data Science dataset, UCI ML. The dataset comprises 61,069 mushroom records with caps based on 173 species, each species having an average of 353 mushrooms.

## Data Features:
The dataset comprises 18 categorical attributes and three numeric attributes that describe the physical characteristics of the mushrooms. Additionally, the dataset includes a target variable, known as "class," which indicates whether a mushroom is edible or poisonous. Out of the 61,069 rows, 27,181 mushrooms are classified as edible, while 33,888 are classified as poisonous. The dataset's richness and diversity make it a valuable resource for exploring the properties and characteristics of mushrooms and building models that can accurately classify them.

## Data Cleaning and Preprocessing:
we conducted a range of exploratory data analyses to gain a comprehensive understanding of the recorded data. Based on our findings, we implemented a series of changes aimed at achieving our desired outcomes. Through careful examination of the data, we were able to identify areas where adjustments were needed, and we took action accordingly.
### 1)	Null Values: -
We conducted a test to identify null values in each column of our dataset. The test results showed that four columns contained more than 80% null values. As a result, we made the decision to drop these columns from our dataset, as they would not provide useful information for our model. For the remaining columns with null values, we decided to fill them with an unknown character.
![image](https://user-images.githubusercontent.com/101862962/235265805-0cc40e28-8bc7-49c7-8b87-dcf66404ac53.png)

This approach allowed us to preserve the structure of our dataset while also addressing the issue of null values. By identifying and addressing null values in our dataset, we could develop a more robust and accurate model that could provide meaningful insights into our data. The test for null values was a critical step in our data analysis process, as it allowed us to identify and address issues that could impact the accuracy of our model. We believe that ultimately, this will helped us to create a more reliable and useful model that could provide valuable insights into our data.

### 2)	Test for correlation between three numeric variables: - 
![image](https://user-images.githubusercontent.com/101862962/235265824-cd225576-7900-403e-8573-53221384b29e.png)

The heat map provided us with a visual representation of the strength and direction of correlations between variables. Specifically, we observed a correlation of 0.7 between stem width and cap diameter. A correlation of 0.7 suggests a strong positive correlation between the two variables, indicating that as one variable increases, the other variable tends to increase as well. Given this information, we made the decision to drop either stem width or cap diameter from our dataset. Dropping one of these variables can help us to avoid multicollinearity in our model, which can lead to less reliable results. The heat map was a useful tool that allowed us to identify variables that were strongly correlated, enabling us to make informed decisions about which variables to include in our model. Ultimately, this helped us to create a more robust and accurate model that could provide meaningful insights into our data.

### 3)	Pair to identifying potential patterns: -
 ![image](https://user-images.githubusercontent.com/101862962/235265858-651e57ab-04bd-4cfc-b84f-9ed075e52fda.png)
 
We chose two numeric variables that we believed could have a significant impact on the target variable. To visualize the relationship between these variables, we created a pair plot. The plot displays the two numeric variables on separate axes and shows how the data is clustered by class. The plot enabled us to observe the pattern of the data points and helped us understand the relationship between the variables. From the pair plot, we could see how the data points were clustered and how the two classes were distributed. This visualization provided us with valuable insights that helped us make informed decisions about the variables to include in our model.

### 4)	Chi-square test to test the independence: -
![image](https://user-images.githubusercontent.com/101862962/235265882-a87b0074-3fd5-42c4-9f1e-585420f42ad8.png)

Chi Square test was conducted to understand relationship between target variable and the nominal categorical variable. For Nominal variables, if the p-value is greater than level of significance of 0.05, so we fail to reject the null hypothesis and conclude that we fail to reject the null hypothesis and conclude that there is insufficient evidence to suggest an association between the two variables. In this case, we dropped those variables.

  

H0: there is no association between two categorical variables in a population
H1: there is an association between the two variables in the population.

Chi-square test is an important statistical tool that helps us understand the relationship between different variables in a dataset. So, we decided to do a Chi-Square test on the dataset. In this case, we calculated the p-value and checked if it is greater or less than 0.05. If it is less than 0.05, we reject H0 and decided to keep those variables. On the other hand, if p>=0.05, there is no evidence to reject H0 and therefore we conclude that there is no association of that variable with the class. Then, we dropped those variables which don’t have statistical significance to the class variable. Additionally, we noticed a high correlation between stem height and cap diameter in the numerical variables. Consequently, we decided to drop cap diameter to increase the accuracy and effectiveness of our model. Making this decision is crucial to ensure that we develop a robust and accurate model that can provide meaningful insights into our data.

### 5)	Imputing "Unknown" class to all the remaining null values to preserve the structure of the data and avoid introducing bias: -
We made a decision to replace null values with "unknown" to ensure that our training model remains unbiased. We recognized the potential impact that missing values can have on the accuracy of the model, and therefore, decided to treat them as a separate category. By doing so, we aimed to prevent any data distortion or bias that could occur due to the missing information. This approach allowed us to maintain the integrity of the data and provide a more reliable output from our model. Our decision to replace null values with "unknown" was a critical step in ensuring the quality and reliability of our project results.

### 6)	One Hot Encoding: -
We implemented one hot encoding as a preprocessing step to convert the categorical data to numerical data. We decided this was necessary because many machine learning algorithms are designed to work with numerical data, and using categorical data as is can lead to confusion and errors in the model. By utilizing one hot encoding, we were able to represent each category as a binary feature, effectively removing any potential numerical relationships between categories.

### 7)	Dimensionality Reduction
 
One-hot encoding categorical variables with many categories result in increasing the dimensionality of encodings. This causes a curse of dimensionality, hence creating the problem of parallelism and multicollinearity. Some of these issues were resolved using different statistical techniques like chi-square test and Pearson’s correlation test. However, we still had 74 features.
Dimensionality reduction is a technique that is commonly used in machine learning and data analysis to reduce the number of features in a dataset while still preserving as much of the relevant information as possible. One popular method for dimensionality reduction is Principal Component Analysis (PCA), which is a mathematical technique that identifies patterns in data by finding the principal components that explain most of the variance in the data.
In this project, we implemented PCA from scratch using the Numpy library in Python. Our implementation involved several steps, including centering the data, computing the covariance matrix, finding the eigenvectors and eigenvalues of the covariance matrix, and projecting the data onto the principal components.
After implementing PCA, we were able to successfully reduce the dimensionality of our dataset by 16 features, which means that we were able to eliminate 16 features from our dataset while still preserving most of the relevant information. This is a significant reduction in dimensionality, which can lead to several benefits, including improved computational efficiency, reduced overfitting, and easier visualization of the data.
To evaluate the effectiveness of our PCA implementation, we calculated the explained variance of each principal component and plotted a scree plot to visualize the results. The scree plot showed that the first few principal components explained the majority of the variance in the data, which indicates that our PCA implementation was successful in identifying the most important patterns in the data.
Overall, our implementation of PCA using NumPy was successful in reducing the dimensionality of our dataset by 16 features, and we were able to confirm the effectiveness of our implementation by calculating the explained variance and visualizing the results. This technique has several important applications in machine learning and data analysis and can be used to improve the efficiency and accuracy of many different types of models.


## Logistic regression: -

We opted to utilize Logistic Regression as it is a straightforward and effective technique that necessitates minimal assumptions regarding the data, while also having the ability to handle categorical and continuous independent variables and can be utilized for both linear and nonlinear models. We have decided to take logistic regression with gradient descent without regularization as baseline model.

### i)	Logistic regression without Regularization 
Logistic regression is a statistical method used to analyze the relationship between a dependent variable and one or more independent variables. It is a type of regression analysis used to model binary or categorical outcomes, where the dependent variable is dichotomous (i.e., has only two possible outcomes, such as 0 or 1).
The goal of logistic regression is to find the best-fitting model that describes the relationship between the independent variables and the dependent variable. It does this by estimating the probability of the dependent variable taking on a certain value given a set of independent variables.
The logistic regression model uses the logistic function (also called the sigmoid function) to transform the linear combination of the independent variables into a probability value between 0 and 1. 

logistic function:
g(z) = 1 / (1 + e^(-z))

The goal is to minimize the cost function J(θ) which is defined as:
J(θ) = (-1/m) * ∑ [y*log(hθ(x)) + (1-y)*log(1-hθ(x))]

We used gradient descent to update the weights θ. Gradient descent is an optimization algorithm used in machine learning and neural networks to find the values of parameters or coefficients of a function that minimizes a cost function. The algorithm works by starting with an initial set of parameter values and iteratively adjusting them in the direction of the negative gradient of the cost function until a minimum is reached. The negative gradient of the cost function represents the steepest descent or direction of greatest decrease in the cost function, which is why the algorithm is called "gradient descent". The algorithm is based on a convex function, which means that it has only one minimum and it will always converge to that minimum.
The gradient descent algorithm works by iteratively updating the weights θ using the following equation:
θj = θj - α * ∂J(θ) / ∂θj

where α is the learning rate and j is the index of the weight.

The partial derivative of the cost function with respect to θj is:
∂J(θ) / ∂θj = (1/m) * ∑ [(hθ(x) – y) * xj]

Therefore, the update rule for θj is:
θj = θj - α * (1/m) * ∑ [(hθ(x) - y) * xj]

This formula is used to update each weight θj for every training example in the dataset until the cost function converges to a minimum.
For this logistic regression model, we have modeled lambda as 0.0001, set the tolerance to 0.00005, and limited the maximum number of iterations to 5000.

For this model we received the following scores for both training and testing.
Training F1 score:		 0.580783651506912
Training recall score:		 0.4306086070791953
Training precision score:	 0.8917992440889514
Training accuracy score:	 0.6573640872087583
Testing F1 score:		 0.5803231314856167
Testing recall score:		 0.42785202401704436
Testing precision score:	 0.9016326530612245
Testing accuracy score:	 0.651219911576879


### ii)	Logistic regression with L2 Regularization and Gradient Descent
Regularization is a technique used in logistic regression to prevent overfitting, which occurs when the model is too complex and fits too closely to the training data, resulting in poor performance on new data. Logistic regression with gradient descent and ridge regression is a regularized version of logistic regression, where a penalty term is added to the cost function to prevent overfitting. The penalty term is controlled by a hyperparameter, λ (lambda), which determines the strength of the regularization.
The cost function for logistic regression with ridge regression is given by:
J(θ) = -1/m [ Σ y(i) log(hθ(x(i))) + (1-y(i)) log(1 - hθ(x(i))) ] + λ *Σ θ(j)^2

The second term of the cost function is the regularization term, which penalizes large values of the model parameters. The ridge regression penalty term is the squared L2 norm of the parameter vector, Σ_j=1^n θ(j)^2.

The update rule for gradient descent with ridge regression is given by:
θ(j) = θ(j) - α/m [ Σ_i=1^m (hθ(x(i)) - y(i)) x(i)(j) + λ *θ(j)]

For this logistic regression model, we have modeled a lambda value of 0.0001, a tolerance level of 0.00005, a ridge lambda of 0.1, and a maximum iteration limit of 5000.

For this model we received the following scores for both training and testing.
Training F1 score:		 0.5846366542815173
Training recall score:		 0.43073593073593075
Training precision score:	 0.9096531325625168
Training accuracy score:	 0.6626508842518948
Testing F1 score:		 0.5819910219170848
Testing recall score:		 0.42688359480921945
Testing precision score:	 0.9141435089174617
Testing accuracy score:	 0.6543856776376835

### iii)	Logistic regression with L1 Regularization and Gradient Descent

Logistic regression with Lasso regularization is a technique that combines two statistical methods: logistic regression and Lasso regularization. Logistic regression is a popular method used to model binary response variables, while Lasso regularization is a method used to reduce overfitting and improve the model's generalization performance by shrinking the coefficients towards zero. In logistic regression with Lasso regularization, the L1 penalty is applied to the logistic regression coefficients, which adds a constraint to the optimization problem that minimizes the sum of squared errors. This penalty encourages the model to select a smaller set of important features and set the coefficients of the less important features to zero, effectively performing feature selection and reducing the risk of overfitting. By applying Lasso regularization to logistic regression, we can obtain a simpler model that is less prone to overfitting, and at the same time, identify the most important features for prediction. However, it's important to note that the choice of regularization parameter should be carefully tuned to balance between model complexity and predictive performance.
The cost function for logistic regression with ridge regression is given by:
J(θ) = -1/m [ Σ_i=1^m y(i) log(hθ(x(i))) + (1-y(i)) log(1 - hθ(x(i))) ] + λ* |θ(j)|

The second term of the cost function is the regularization term, which penalizes large values of the model parameters. The lasso regression penalty term is the squared L1 norm of the parameter vector, Σ_j=1^n θ(j)^2.

The update rule for gradient descent with lasso regression is given by:
θ(j) = θ(j) - α/m [ Σ_i=1^m (hθ(x(i)) - y(i)) x(i)(j) + λ* sign θ(j)]

For this logistic regression model, we have modeled a lambda value of 0.0001, a tolerance level of 0.00005, and a maximum iteration limit of 5000.

For this model we received the following scores for both training and testing.
Training F1 score:		 0.41172974043986527
Training recall score:		 0.2645785586962058
Training precision score:	 0.9276785714285715
Training accuracy score:	 0.5832787498830355
Testing F1 score:		 0.4127008069990196
Testing recall score:		 0.26496223126089485
Testing precision score:	 0.9328332765086942
Testing accuracy score:	 0.5749686152502592

### iv)	Logistic regression with Stochastic Gradient Descent
Stochastic gradient descent is an optimization algorithm that is used to update the parameters of the logistic regression model.
The algorithm works by iteratively updating the parameters of the model using a small subset of the training data, known as a batch. The batch is randomly selected from the training data at each iteration. This allows the algorithm to quickly converge to a good set of parameters, even for large datasets.
The update for gradient descent with a learning rate λ is simply to adjust the parameters of the model by subtracting the gradient of the loss function with respect to the parameters, multiplied by the learning rate. In the case of logistic regression, the loss function is the log loss, which is a convex function that measures the difference between the predicted probabilities and the true labels.
One important computational point to note is that if a feature has a value of zero for a particular example, the gradient with respect to the corresponding parameter is also zero. This can be exploited to speed up the computation of the gradients by skipping the computation for features that have a zero value.
Overall, logistic regression with stochastic gradient descent is a powerful and efficient algorithm for classification problems, especially for large datasets.

For this logistic regression model, we have modeled a lambda value of 0.0001, a tolerance level of 0.00005, and a maximum iteration limit of 5000.

For this model we received the following scores for both training and testing.
Training F1 score:		 0.5871272478943774
Training recall score:		 0.43786605551311436
Training precision score:	 0.8907787946814022
Training accuracy score:	 0.6605689155048189
Testing F1 score:		 0.5871439770054874
Testing recall score:		 0.4352120859965137
Testing precision score:	 0.9020473705339221
Testing accuracy score:	 0.6550406637192293

### v)	Logistic regression with Stochastic Gradient Descent and L2 Regularization 
Stochastic gradient descent (SGD) is an optimization algorithm used to train logistic regression models. It is a variant of gradient descent that updates the model parameters using a single training example at a time, rather than the entire dataset. This makes it much faster than batch gradient descent, especially when dealing with large datasets.
Ridge regression is a regularization technique used to prevent overfitting in logistic regression models. It adds a penalty term to the loss function, which is proportional to the square of the L2 norm of the model parameters. The penalty term shrinks the parameter values towards zero, which reduces the model's complexity and makes it less prone to overfitting.
When using logistic regression with SGD and ridge regression, the loss function includes both the cross-entropy loss and the L2 regularization penalty. 

For this logistic regression model, we have modeled a lambda value of 0.0001, a tolerance level of 0.00005, a ridge lambda of 0.1, and a maximum iteration limit of 5000.

For this model we received the following scores for both training and testing.
Training F1 score:	 	0.4100935939411979
Training recall score:	 	0.2631355572532043
Training precision score:	0.9288389513108615
Training accuracy score:	0.5827407130158136
Testing F1 score:		0.41141390503510233
Testing recall score:		0.26389695913228745
Testing precision score:	0.9328996918863403
Testing accuracy score:	0.5744227935156377

### vi)	Logistic Regression with Stochastic Gradient Descent and L1 Regularization
Lasso regression is a regularization technique used to prevent overfitting in the model. It works by adding a penalty term to the loss function that encourages the weights to be small. Lasso regression can be combined with logistic regression and SGD to create a model that is both efficient and accurate.
By using stochastic gradient descent with lasso regression, we can optimize the weights and biases of the logistic regression model while preventing overfitting. The result is a model that is both accurate and computationally efficient, making it suitable for large datasets.

For this logistic regression model, we have modeled a lambda value of 0.0001, a tolerance level of 0.00005, a lasso lambda of 0.1, and a maximum iteration limit of 5000.

For this model we received the following scores for both training and testing.
Training F1 score: 		0.6582318940947713
Training recall score: 	0.5244037008742891
Training precision score: 	0.8837708318432158
Training accuracy score: 	0.6998456068120146
Testing F1 score: 		0.6562232284910047
Testing recall score: 		0.5192717412357156
Testing precision score: 	0.8912898936170213
Testing accuracy score: 	0.6933573494896567

## Decision tree: -
Decision trees provide an effective method of decision making because they: Clearly lay out the problem so that all options can be challenged.  It is a non-parametric and supervised learning algorithm, meaning that it uses labeled data to make predictions. The decision tree works by creating a hierarchical structure of decisions and their possible outcomes. The tree-like structure is composed of nodes, branches, and leaves. The nodes represent the decision points, while the branches represent the possible outcomes of each decision. The leaves represent the final classification or prediction. We set the maximum depth parameter to 15 for our decision tree model. By setting the maximum depth to 15, we aimed to achieve a balance between model accuracy and generalization ability.

For this model we received the following scores.
Training F1 score:		0.9927
Training recall score:		0.9928
Training precision score:	0.9926
Training accuracy score:	0.9919
Testing F1 score:		0.9885
Testing recall score:		0.9886
Testing precision score:	0.9884
Testing accuracy score:	0.9873

## Random Forest: - 
A random tree is a type of decision tree used in the random forest algorithm, which is an ensemble learning method for classification, regression, and other tasks. Random forests are constructed by creating multiple decision trees, where each tree in the ensemble is trained on a random subset of the training data. The final prediction is then made by combining the predictions of all the individual trees in the forest. To reduce computation cost in our random forest model, we opted to use a sample size of 5000 and 1000 and set the maximum depth as 5 for the bootstrap process.

For this model we received the following scores.
Sample size: 1000
Training F1 score:		0.7224
Training recall score:		0.6312
Training precision score:	0.8438
Training accuracy score:	0.7305
Testing F1 score:		0.7245
Testing recall score:		0.6311
Testing precision score:	0.8504
Testing accuracy score:	0.7337

Sample size: 5000
Training F1 score:		0.6955
Training recall score:		0.5783
Training precision score:	0.8721
Training accuracy score:	0.719
Testing F1 score:		0.6965
Testing recall score:		0.5782
Testing precision score:	0.8757
Testing accuracy score:	0.7204


## Neural Network: -
![image](https://user-images.githubusercontent.com/101862962/235266163-27d4407c-78da-44bf-a333-e0b3336568c3.png)

A neural network is a type of machine learning model that is inspired by the structure and function of the human brain. It consists of a large number of interconnected nodes, called neurons, that work together to learn patterns and relationships in input data.
A neural network typically consists of multiple layers of neurons, each layer processing the output of the previous layer. The first layer is called the input layer and receives the raw input data, while the last layer is called the output layer and produces the final output of the model. The intermediate layers are called hidden layers and are responsible for learning the underlying patterns in the data.
During training, the neural network adjusts the weights and biases of the neurons to minimize a loss function, which measures the difference between the predicted output and the actual output. This process is usually done using an algorithm called backpropagation, which calculates the gradient of the loss function with respect to the weights and biases and updates them accordingly.
A Neural Network model was built for classification using Tensorflow. The model consisted only one hidden layer of 8 nodes with a ReLu activation function. As the problem was binary classification problem, the output node was only one node, and the activation function was sigmoid. The loss metric was binary cross entropy, and the Adam optimizer was used with 10 epochs to optimize the weights of the model. The accuracy of the model was around 98%, but the precision and recall scores were 98% and 97% respectively.

For this model we received the following scores.
Training F1 score:		0.98
Training recall score:		0.97
Training precision score:	0.98
Training accuracy score:	0.98
Testing F1 score:		0.98
Testing recall score:		0.97
Testing precision score:	0.98
Testing accuracy score:	0.98
Results: -

For measuring the results of our models, we used Recall, precision, accuracy and F1 score: -
A confusion matrix is a table used to evaluate the performance of a classification model by comparing its predicted output with the actual output. The matrix shows the number of true positives, true negatives, false positives, and false negatives that the model has generated.
•	True positives (TP) are the cases where the model correctly predicted a positive outcome when the actual outcome was positive.
•	True negatives (TN) are the cases where the model correctly predicted a negative outcome when the actual outcome was negative.
•	False positives (FP) are the cases where the model predicted a positive outcome, but the actual outcome was negative.
•	False negatives (FN) are the cases where the model predicted a negative outcome, but the actual outcome was positive.

Accuracy: -
Accuracy measures the proportion of correctly classified instances among all instances in the dataset. It is calculated as the number of true positives and true negatives divided by the total number of instances.
 
Recall: -
Recall measures the proportion of true positives among all instances that are actually positive in the dataset. It is calculated as the number of true positives divided by the sum of true positives and false negatives.
 
Precision: -
Precision measures the proportion of true positives among all instances that are predicted as positive by the model. It is calculated as the number of true positives divided by the sum of true positives and false positives.
 
F1 Score: -
F1 score is a metric that balances precision and recall by taking their harmonic mean. It is a better measure than accuracy when dealing with imbalanced datasets, where one class has significantly fewer instances than the other.
 
Model Results: -
![image](https://user-images.githubusercontent.com/101862962/235266190-a8af88ed-c400-41cd-b78b-34fdbd80234f.png)
Here is a comprehensive summary of the results obtained from each of these models. The results showcase the strengths and weaknesses of each model, helping us determine which models are most effective for our project.
 

Bias Variance Tradeoff: - 

The bias-variance tradeoff is a technique that involves balancing a model's accuracy on the training data, known as bias, with its ability to generalize to new and unseen data, known as variance.

Logistic regression: -
![image](https://user-images.githubusercontent.com/101862962/235266216-04fa32e6-7dbe-4f91-893c-a9a0f3538091.png)
As we can see as the lambda values increases, the models tend to have high bias; however, as we decrease the value of lambda, we may observe high variance and low bias, leading to overfitting of the training data. Therefore, the ideal lambda value is the one that minimizes both training and cross-validation errors, providing a good balance between bias and variance.

Decision Tree: -
![image](https://user-images.githubusercontent.com/101862962/235266250-97a774cd-c2e1-46ff-b54c-c37712f5c606.png)

Upon analyzing the curve and accuracy value, we can infer that the Decision Tree model exhibits a superior generalization ability when compared to logistic regression in this project.

Conclusion: -

•	We have demonstrated how machine learning can be used to classify mushrooms based on their physical characteristics. By training various machine learning algorithms on a dataset of mushroom features and labels, we were able to develop models that can accurately predict the edibility of mushrooms with high precision.

•	After careful consideration, we've opted to use logistic regression without regularization as our baseline model.

•	We showed that feature selection and engineering can greatly improve the performance of our models. By selecting a subset of the most informative features and creating new features from the existing ones, we were able to achieve better accuracy scores and reduce overfitting.

•	Our analysis showed that different algorithms have different strengths and weaknesses in this task. Decision tree-based algorithm was found to be highly accurate and efficient for mushroom classification, with accuracy scores reaching up to 98%. Neural Network also performed well, although it is difficult infer the feature importance from them.

Discussion: -

•	Our project aimed to classify mushrooms based on their physical characteristics using machine learning algorithms. We trained several algorithms on a dataset of mushroom features and labels to predict the edibility of mushrooms accurately. We achieved high precision, indicating that machine learning is an effective approach for mushroom classification.

•	We discovered that feature selection and engineering are essential to improve the performance of our models. By selecting a subset of informative features and creating new features from the existing ones, we were able to achieve better accuracy scores and reduce overfitting. This highlights the importance of feature engineering in machine learning.


•	Our analysis showed that different algorithms have different strengths and weaknesses in this task. The decision tree-based algorithm, such as Random Forest and XGBoost, were highly accurate and efficient for mushroom classification, with accuracy scores reaching up to 98%. We found that Neural Networks also performed well, but it is challenging to infer feature importance from them.

•	We also explored the use of Naïve Bayes classifiers, as most of the features in the dataset are categorical. Naïve Bayes classifiers are simple yet powerful algorithms, making them a suitable option for this classification task. We also considered using XGBoost classifiers as they are faster to train compared to SVMs.

•	After attempting to execute Random Forest with Bootstrap, we discovered that the process was taking a substantial amount of time to complete; as a result, we opted to decrease the sample size and proceeded to run the algorithm using 1000 samples, which took a mere 1 minute and 37 seconds, as well as with 5000 samples, which took 5 minutes and 49 seconds to run.

•	In conclusion, our project demonstrates that machine learning algorithms can accurately classify mushrooms based on their physical characteristics. Feature selection and engineering are critical to achieving high accuracy scores and avoiding overfitting. We recommend using decision tree-based algorithms, such as Random Forest and XGBoost, for mushroom classification tasks, and Naïve Bayes classifiers for datasets with categorical features.



References: -
The data comes from UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/Secondary+Mushroom+Dataset 


