Alexandra Gómez Villa
0 minutes 5 seconds0:05
Alexandra Gómez Villa 0 minutes 5 seconds
OK, so today we are going to begin to one of the most powerful models that you can learn to classify or or do regression and that is super vector machines so.
Alexandra Gómez Villa 0 minutes 23 seconds
So a little as a small introduction is that before.
Alexandra Gómez Villa 0 minutes 30 seconds
2012 it was, which actually was the the beginning of the deep learning era. Everything was done using a super vector machines like every state-of-the-art machine learning pipeline basically consisted of some features that.
Alexandra Gómez Villa 0 minutes 50 seconds
Some experts select either if you are of course if you are for instance in the in the medical field it will be some doctors, some some research in medicine and then you usually extract features from images from the time series analysis for whatever signal that you are analyzing.
Alexandra Gómez Villa 1 minute 10 seconds
And then after you do proper feature selection, then you need to select the model. So the model that that was always there in any state-of-the-art, state-of-the-art means like the most the the the highest accuracy models.
Alexandra Gómez Villa 1 minute 25 seconds
There is the at that time before 2012 it was always a super vector machine, so.
Alexandra Gómez Villa 1 minute 36 seconds
That's that's why it's still really, really useful if you need a classifier to use a super vector machine. And actually my first advice and something really, really important that you should remember from this class is that if you have any.
Alexandra Gómez Villa 1 minute 56 seconds
Machine learning problem and you already have the features. The best option, the best classifier to try first. If it's a supervised problem, it's a super vector machine. It's really, really good.
Alexandra Gómez Villa 2 minutes 10 seconds
OK, so let's try to dip in to try to understand why it's so good.
Alexandra Gómez Villa 2 minutes 16 seconds
Um, so yeah, here is a little bit the outline. We are going to discuss, remember a little bit about the bias, variance trade off because that that concept is really important in super vector machine. Then we are going to discuss about maximal margin classifier, which is the.
Alexandra Gómez Villa 38 minutes 39 seconds
And here you can see for kernel all the options that you can have, right? Linear, which is basically not using any kernel, polynomial, radial basis function with X which is Gaussian, sigmoid and pre-computed. So you can actually develop.
Alexandra Gómez Villa 38 minutes 57 seconds
your own kernel function. Of course, this is if you are really familiar with your problem and it's not so trivial because it needs to to fulfill some mathematical conditions, but you can put your kernel here and Python is going to use your kernel.
Alexandra Gómez Villa 39 minutes 13 seconds
Then in case you use a polynomial kernel, you play with here with the degree. So remember I told you the first option is always to try with polynomial. You try polynomial. Once you choose polynomial, you need to play with this degree, which is another hyperparameter. So you begin with degree two, then degree three and so forth.
Alexandra Gómez Villa 39 minutes 33 seconds
And so on. Usually something bigger that degrees 567 is just is. It just doesn't make a lot of sense, but usually try 4 ranges between 2:00 and 7:00. It's something reasonable here.
Alexandra Gómez Villa 39 minutes 50 seconds
Then in case you choose the Gaussian kernel, uh this will be kind of important, the gamma, umm which is basically how you are going to scale the features of the kernel. Usually you don't change this
Alexandra Gómez Villa 40 minutes 10 seconds
value a lot, but it's possible to play a little bit if you want to increase a little bit the accuracy.
Alexandra Gómez Villa 40 minutes 18 seconds
Stop.
Alexandra Gómez Villa 40 minutes 20 seconds
So summary. Um, yes.

Johanna Ursula Albers
40 minutes 22 seconds40:22
Johanna Ursula Albers 40 minutes 22 seconds
So.
AV
Alexandra Gómez Villa
40 minutes 28 seconds40:28
Alexandra Gómez Villa 40 minutes 28 seconds
I think someone.

Johanna Ursula Albers
40 minutes 28 seconds40:28
Johanna Ursula Albers 40 minutes 28 seconds
What range would you choose to see?
Johanna Ursula Albers 40 minutes 30 seconds
You should see me.
AV
Alexandra Gómez Villa
40 minutes 31 seconds40:31
Alexandra Gómez Villa 40 minutes 31 seconds
Sorry, sorry, can you repeat? Because you you're you you got, you got.
Alexandra Gómez Villa 40 minutes 41 seconds
I think I cannot hear you.
Alexandra Gómez Villa 40 minutes 53 seconds
Um.
Alexandra Gómez Villa 40 minutes 58 seconds
Who made that question? Because I think I lost you.
Alexandra Gómez Villa 41 minutes 4 seconds
OK.

Iris Mestres Pascual
41 minutes 5 seconds41:05
Iris Mestres Pascual 41 minutes 5 seconds
I I think so. Just let the mic on. It wasn't a question, just someone that didn't mute.
AV
Alexandra Gómez Villa
41 minutes 13 seconds41:13
Alexandra Gómez Villa 41 minutes 13 seconds
Uh, OK.
Alexandra Gómez Villa 41 minutes 17 seconds
Thanks. Um.
Alexandra Gómez Villa 41 minutes 20 seconds
Stop.
Alexandra Gómez Villa 41 minutes 23 seconds
So as a summary and.
Alexandra Gómez Villa 41 minutes 28 seconds
Here you will notice that we are missing a lot of mathematical details about support vector machines, and this is in part of purpose. So the mathematics of the kernel trick, how to find the proper kernel is.
Alexandra Gómez Villa 41 minutes 46 seconds
Like more advanced level is really out of the scope of this course, so we are not discussing how to find kernels. What is the weight weight of use kernel here? Just it's enough if you learn that kernel is a transformation that you use to transform your features.
Alexandra Gómez Villa 42 minutes 2 seconds
That's that's that's more than enough.
Alexandra Gómez Villa 42 minutes 6 seconds
Um, but apart from that for yes.

Iris Mestres Pascual
42 minutes 8 seconds42:08
Iris Mestres Pascual 42 minutes 8 seconds
Let's turn it up.
Iris Mestres Pascual 42 minutes 10 seconds
It it turns out that it was a question, but the mic didn't work. So the question is in which range would you choose the C parameter?
AV
Alexandra Gómez Villa
42 minutes 17 seconds42:17
Alexandra Gómez Villa 42 minutes 17 seconds
Yes.
Alexandra Gómez Villa 42 minutes 22 seconds
Uh, OK, so.
Alexandra Gómez Villa 42 minutes 25 seconds
This really depends on the problems guys. I will tell you for experience in general this parameter C usually is between range one and 9 for me.
Alexandra Gómez Villa 42 minutes 41 seconds
I have applied support vector machines in many kind of domains and usually is something less than five usually, but please just try that range. So don't don't go and say hey my machine learning teacher said that this enough if I go from from from wine to.
Alexandra Gómez Villa 43 minutes 1 second
6 No, that's incorrect because like those are my my problems that the domain problems that I try. Perhaps you are in a really weird topic in a really different problem. So the the like the most smart thing to do is you program a script on automatic.
Alexandra Gómez Villa 43 minutes 20 seconds
script writing Python. And that script, basically you put a for loop. In that for loop, you put a vector of many values of C. C, 1, 2, 3, 4, till 10. Then inside that for loop, you train your super vector machine as as
Alexandra Gómez Villa 43 minutes 40 seconds
actually this, something like this, and then at the end of the training, just before after this, you pull here a cross validation.
Alexandra Gómez Villa 43 minutes 52 seconds
And that cross validation give you the the mean and the standard deviation accuracy with that value of C. Then the for loop is going to go again and change the value of C and after you finish that what you are going to have is actually a report.
Alexandra Gómez Villa 44 minutes 9 seconds
a vector in which you can print for each value of C what's the best accuracy for your problem. So that's the useful practice to do.
Alexandra Gómez Villa 44 minutes 18 seconds
Begin with C between 1:00 and 10:00 and you see that and that's more than enough, of course.
Alexandra Gómez Villa 44 minutes 27 seconds
If you can do that for loop with changing the value of C also changing, then you can put another for loop to change the the kernel right to another for loop here with the kernel Gaussian and other kernel polynomial. That's really good if you can do all of that test.
Alexandra Gómez Villa 44 minutes 45 seconds
If you have a lot of data, so I am speaking about really, really high quantities of data, millions of samples then.
Alexandra Gómez Villa 45 minutes
This is not so easy to do. Then you cannot do that for loop because you are going to need a lot of computation. Then you need to do a smarter things, choose a a subset of your data and so forth and so on, but perhaps.
Alexandra Gómez Villa 45 minutes 16 seconds
I am. I am. I am complicating things enough. Let's say for this course and for the quantities of data of the usual machine learning problems, you put a for loop. In the for loop you put several values of C you check with cross validation.
Alexandra Gómez Villa 45 minutes 32 seconds
And that's enough.
Alexandra Gómez Villa 45 minutes 35 seconds
OK, so about this key concept guys for me. So more than the math for me, it's important that you understand this stuff. First that super vector machines are solved using a quadratic optimization problem. So basically.
Alexandra Gómez Villa 45 minutes 55 seconds
You are solving you are solving the super vector machine cost function in the same way as a logistic regression. Then this is yeah, this is basically as as super vector machines is using a linear linear classifiers. You can introduce these kernels.
Alexandra Gómez Villa 46 minutes 12 seconds
And then the kernels are allowing us to classify these nonlinear separable data sets.
Alexandra Gómez Villa 46 minutes 21 seconds
The kernel functions allow us to go from lower dimensions to really high dimension in which our data is separable.
Alexandra Gómez Villa 46 minutes 30 seconds
And we have some slack variables, the the variables which are important, those support vectors that are important to make this margin classifier this this decision boundary more flexible or not.
Alexandra Gómez Villa 46 minutes 45 seconds
Yeah, so here that we need to adjust in training always, always the kernel and the C.
Alexandra Gómez Villa 46 minutes 57 seconds
OK, so apart from that, yeah, so you as always you have the the Python notebook in which in which you can take a look of how it is implemented. There is a nice example with faces at the end.
Alexandra Gómez Villa 47 minutes 13 seconds
So it's more a more real case too. And my final words on this is I'm just going to repeat the same at the beginning.
Alexandra Gómez Villa 47 minutes 22 seconds
This is probably the best classifier of all the course.
Alexandra Gómez Villa 47 minutes 28 seconds
So if you have a machine learning problem, a supervised machine learning problem, this is the first model that you should always try a super vector machine.
Alexandra Gómez Villa 47 minutes 42 seconds
OK. So then any question guys?
Alexandra Gómez Villa 47 minutes 49 seconds
No. OK. So then had a good weekend and see you on Wednesday. Yes.

Delia Clara Podaru Savu
47 minutes 56 seconds