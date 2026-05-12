Alexandra Gómez Villa
0 minutes 3 seconds0:03
Alexandra Gómez Villa 0 minutes 3 seconds
Today is that we begin. We begin with the second block of our course of the four blocks, and that's basically supervised learning. From here on, you will notice that most of the machine learning models that we.
Alexandra Gómez Villa 0 minutes 21 seconds
See in the course will be basically supervised learning models. This is you will you will see in a second, but this is basically because it's easier to adjust models when you have a target or a ground truth. But OK, but let's let's dive in.
Alexandra Gómez Villa 0 minutes 38 seconds
So you can get the point.
Alexandra Gómez Villa 0 minutes 42 seconds
OK, so so this is all the data for today. We are going to check a supervised learning, the definition, the two types of supervised learning target, which is regression and versus classification. Then we are going to check like the most basic.
Alexandra Gómez Villa 0 minutes 58 seconds
A regression base, which is the linear regression. We are going to see motivation to possible ways to solve this linear regression problem with least squares or gradient descent. And then at the end we are going to talk a little bit.
Alexandra Gómez Villa 1 minute 14 seconds
About bias and variance trade off and a cross validation.
Alexandra Gómez Villa 1 minute 22 seconds
So here is our our machine learning pipeline. We are focusing on machine learning selection. Let's diving into supervised learning. So in order to talk about machine learning, remember we have previously talked about.
Alexandra Gómez Villa 1 minute 38 seconds
Unsupervised learning, right? So in unsupervised learning, we say, OK, we have this set of features, this weight, width and color for each sample that we have. But we really don't know that the target. We don't know the the true label of each one of these samples. We don't know which which fruit corresponds each one. And we say, OK.
Alexandra Gómez Villa 1 minute 58 seconds
OK, what we are going to do is just to perform some kind of cluster of grouping based on some features and then we will gonna end up with something like this, right? So we're going to have these nice groups and this group we are going to assign then a label and say, OK, I don't know what is this, but this is the thing one.
Alexandra Gómez Villa 2 minutes 17 seconds
If we are doing fruits, we will say this is the fruit one and this is going to be the fruit two. And with this it's really useful because then if you don't know what you are looking for, you can have you can find patterns in your data. For instance, for market segmentation you can find patterns about which kind of.
Alexandra Gómez Villa 41 minutes 44 seconds
Basically it's a line that perhaps is not completely following the pattern on my data. Perhaps it will be it is more close to my data following this cure or following a cure line like this which is following each single point in my data.
Alexandra Gómez Villa 42 minutes 4 seconds
But we are ignoring that. We are ignoring the position of all of these individual samples. We just say that no, this simple line, I don't care if it's if it's not following the position of each single point is the one who explains my data.
Alexandra Gómez Villa 42 minutes 23 seconds
So there is a risk here that perhaps this is an error. Perhaps actually this line should have some curves right here or to have another shape which closely follow my data, but I am just oversimplifying everything.
Alexandra Gómez Villa 42 minutes 38 seconds
This is usually called also underfitting because I am not closely following my data. Then the other possible source of error is the variance. This is the other side of the coin. This is trying to exactly match.
Alexandra Gómez Villa 42 minutes 57 seconds
All of my training data, all of my samples, it will mean a cure that is closely following each individual sample.
Alexandra Gómez Villa 43 minutes 7 seconds
This is called also overfitting. It's fitting exactly to my data. The problem with this other with variance is that perhaps in real life when once I trained in my model with my data and try to use it for a real life deployment, perhaps we are.
Alexandra Gómez Villa 43 minutes 27 seconds
Real life data is not closely following my training data. Perhaps I have this cure that is really following my data, but it happens that in real life I received samples that are.
Alexandra Gómez Villa 43 minutes 44 seconds
In this part.
Alexandra Gómez Villa 43 minutes 46 seconds
So since I find a model which is really, really good predicting my training data.
Alexandra Gómez Villa 43 minutes 56 seconds
The problem is that that's really different from reality and that could be the case because either I didn't have enough data, I didn't have a samples from some real real life cases or some kind of conditions.
Alexandra Gómez Villa 44 minutes 12 seconds
And since those samples were not in my training set, they were not memorized and they are considered not to be present and my model is following only my training data. So basically reality is going to be really different from one model what fit.
Alexandra Gómez Villa 44 minutes 28 seconds
So as you can see these two sides of the same coin. So either I assume that my data set is just like a role representation, a kind of representation of what I want to assume and create really simple models that are not following specifically my data but in general terms.
Alexandra Gómez Villa 44 minutes 48 seconds
Or I either I either have a model that really close memorize over fits completely to my to my data set, which basically have the consequence that perhaps once I deploy in a real in real life.
Alexandra Gómez Villa 45 minutes 3 seconds
A real life data is not as the data in my training set. So when you are trying to train a model, you need to balance this off these two parts. That's why it's called a trade off. You need to balance between really overfitting to be really close to your data and be far away enough.
Alexandra Gómez Villa 45 minutes 23 seconds
So you can take into account samples that are not in your training set, something that is not present in your data.
Alexandra Gómez Villa 45 minutes 33 seconds
Speaking of that, um.
Alexandra Gómez Villa 45 minutes 37 seconds
We need.
Alexandra Gómez Villa 45 minutes 40 seconds
To find ways to mitigate the issue of finding in deployment data that is not present in my training set.
Alexandra Gómez Villa 45 minutes 50 seconds
So before remember we talk about the training, training and test split and you have been applying that to your in in the Python notebooks in which we say OK, we usually take our data and we split, we take 70% or something close to that to training and when we.
Alexandra Gómez Villa 46 minutes 10 seconds
When we need to test how good is my model, I always test in a different set which is called the set test set, the 13% that I researched before in order to simulate that real life deployment.
Alexandra Gómez Villa 46 minutes 25 seconds
So I adjust all of my model parameters in the training set and then I check the performance in the test set. If the performance in the test set is really low, it means that I was just memorizing my data set, that the parameters are only good for the training.
Alexandra Gómez Villa 46 minutes 41 seconds
Not for the real life deployment, which again means that I am doing a lot of over fitting and not not enough under fitting the bias try the the variance bias trade off so.
Alexandra Gómez Villa 46 minutes 57 seconds
A really popular validation strategy to test the generalization of my method. That means how good is my method independently of which of which samples I choose to train is called cross validation.
Alexandra Gómez Villa 47 minutes 13 seconds
So in cross validation basically we are going to do the following. Just split the data your all of your data in K faults.
Alexandra Gómez Villa 47 minutes 23 seconds
And then you train your model in the K -, 1 faults and test only in one in in one of those faults then.
Alexandra Gómez Villa 47 minutes 36 seconds
Basically you evaluate the model in the in the one that you leave out and that's the that's the that's the accuracy in that single fall. Then you rotate the falls and repeat evaluating a training in in in K minute one falls and evaluating the one that you you leave out and at the end.
Alexandra Gómez Villa 47 minutes 56 seconds
What is the average performance of all of these individuals that says how good is your model? So with the plot with this diagram is easier to understand. So you have your data set and you divide this for instance here in five folds.
Alexandra Gómez Villa 48 minutes 12 seconds
K equal 5. So then you have these five and you are going to use the blue ones for training your model. Once you're trained your model, you test with the first one and the test and the accuracy that you get here is the performance here. Then you rotate them.
Alexandra Gómez Villa 48 minutes 28 seconds
Now you are going to use the second one for test and the other ones for train and you compute the performance and you rotate again and you rotate again unless until all your faults have been using for training.
Alexandra Gómez Villa 48 minutes 44 seconds
And for testing and then you report the average performance and this is a really great way to evaluate generalization of your model. It will allow you to select parameters, HPER parameters.
Alexandra Gómez Villa 48 minutes 59 seconds
That are really a good fit, a good balance between that variance and bias between that overfitting and underfitting. Because in this way as you can see you are trying all the combinations of your of your data set.
Alexandra Gómez Villa 49 minutes 17 seconds
Problems of co-validation? Well, as you can see here, we are training five times.
Alexandra Gómez Villa 49 minutes 26 seconds
So we are talking about the small data sets as the ones that you are using for this course because of course you are learning the topic. This is not problem at all. You can run this, you can usually you run 1010 faults.
Alexandra Gómez Villa 49 minutes 43 seconds
So you'll need to run your algorithm 10 times, but think about it. Let's say that you're trying to choose the the learning rate, which is the per parameter of your of of your linear regression. So for each value that you are trying for learning rate, you are trying linear rating.
Alexandra Gómez Villa 50 minutes 2 seconds
Equal one, you run this five times and find this value. Then you try living in rate equal 2, then you run again another five. Now learning rate equal 3, another five and so forth and so on. So each time you change the Heaper parameters you need to run five times the algorithm. So this is really costly.
Alexandra Gómez Villa 50 minutes 22 seconds
Computation. Imagine that in a really big data set that is just not feasible. However, when it's possible, this is the best way to try a model to really try how general.
Alexandra Gómez Villa 50 minutes 39 seconds
Is the model and it's always a recommended way. As an example, the current deep learning models, the models that do that that basically you face detection, classification, the state-of-the-art in machine learning models, the deep learning.
Alexandra Gómez Villa 50 minutes 58 seconds
The learning ones use really, really huge data sets. The data sets are so huge and these models are so hard, so computationally costly to train that.
Alexandra Gómez Villa 51 minutes 12 seconds
Is almost. It's really a scarce to find any deep learning paper that use cross validation because it's just not feasible. You cannot spend that quantity of computation because one training could take.
Alexandra Gómez Villa 51 minutes 30 seconds
Days, months even. So it's not feasible to train so many times to find a parameter. So there are other strategies of course, but for not so big data set for the useful data set that I use it in the in in machine learning community in.
Alexandra Gómez Villa 51 minutes 50 seconds
Medical imaging and that kind of stuff is still feasible to use cross validation.
Alexandra Gómez Villa 51 minutes 57 seconds
OK, so.
Alexandra Gómez Villa 52 minutes 1 second
That's for all the lecture part. Any question?
Alexandra Gómez Villa 52 minutes 12 seconds
OK, it's that there is no question. So if you go to the to the campus virtual as always, today we have a Python notebook about linear regression.
Alexandra Gómez Villa 52 minutes 28 seconds
In which basically you are going to check how to do linear regression with gradient descent, with least squares, how the different line change as a function of the learning rate. So in general to practice all what we saw today.
Alexandra Gómez Villa 52 minutes 43 seconds
Remember, 30 of March is the deadline for the assignment.
Alexandra Gómez Villa 52 minutes 51 seconds
And apart from that, if there is no questions, have a great weekend.