Alexandra Gómez Villa
0 minutes 4 seconds0:04
Alexandra Gómez Villa 0 minutes 4 seconds
Okay, it seems that the recording is working. Yeah, please let me know, guys, if you are not seeing the recording in the channel.
Alexandra Gómez Villa 0 minutes 12 seconds
Because if not, like last time, it could be an error in the login or something like that. Okay.
Alexandra Gómez Villa 0 minutes 23 seconds
Okay, so...
Alexandra Gómez Villa 0 minutes 26 seconds
So, good afternoon. The idea today is to begin with, continue with our neural networks topic. So last time we were seeing like pretty general stuff about what is a neural network, what is a multilayer perceptron and how do we train it.
Alexandra Gómez Villa 0 minutes 44 seconds
Today, we are going to do the last on the second part of this, basically going more into details about regularization, about loss functions, about optimizers, and I am trying here to push a little bit more in the practical in the practical sense.
Alexandra Gómez Villa 1 minute 1 second
Okay.
Alexandra Gómez Villa 1 minute 3 seconds
Um...
Alexandra Gómez Villa 1 minute 5 seconds
So, this is the outline. First, I am going to give a really quick recap of what we saw last class, then we are going to discuss about activation functions, which are the best ones, which are used, and then about loss functions for...
Alexandra Gómez Villa 1 minute 21 seconds
binary classification and multi-class classification, then a little bit about optimizers that are used for neural networks, and finally a little bit about regularizations.
Alexandra Gómez Villa 1 minute 33 seconds
The machine, see the machine learning pipeline. Yes, as always, and here is our model pool. Now we are now we are adding neural networks here, right? Last class we saw the first introduction, and we are going to be more into detail. This basically is all the
Alexandra Gómez Villa 57 minutes 35 seconds
And this is a pretty good regularization method, but not so common in a small multilayer perception. So if you go to cycle learn and that kind of stuff, you are not going to find a dropout implemented, but dropout is one of the keys.
Alexandra Gómez Villa 57 minutes 55 seconds
of deep learning architectures. Deep learning architectures are really deep, really plastic, really powerful, and can easily memorize data sets. That's why we need really, really powerful regularization methods, and Dropbox is one of those.
Alexandra Gómez Villa 58 minutes 12 seconds
So, here, just I'm going to show you, this is the multi-layer perception in Sun Q-Learn. You can see here all of those parameters that we saw that we saw today. So, here we have our activation functions. You can see that by default is the relu activation function. Then, here we have the optimizer. You can see again, but by default is...
Alexandra Gómez Villa 58 minutes 34 seconds
I am an optimizer.
Alexandra Gómez Villa 58 minutes 37 seconds
Then we have here some batch size, some learning rate, which here is keep constant.
Alexandra Gómez Villa 58 minutes 46 seconds
So there is no schedule. You can put some schedule here. Then the initial learning rate is here. It's 0.001 because we are using Adam.
Alexandra Gómez Villa 58 minutes 57 seconds
and some things like maximum iterations, random state. Then if we go down, we can continue looking at even more things that we discussed today. We have here the momentum. The hyperparameter of the momentum is here. This is a specific implementation of momentum.
Alexandra Gómez Villa 59 minutes 17 seconds
It's okay, you don't need to look into that. And then you can, you look here, you can find parameters of the add and optimizer. So you have here the beta one and beta two, which most of the time you don't need to change.
Alexandra Gómez Villa 59 minutes 32 seconds
Stop.
Alexandra Gómez Villa 59 minutes 35 seconds
Then, to finalize, to finish the neural network part, then again, why neural networks matter? So, neural networks matter only for one moment in 2012, and it's for the ImageNet classification challenge. So, in 20...
Alexandra Gómez Villa 59 minutes 53 seconds
Then they begin this big competition in which called ImageNet or large scale image recognition challenge using the ImageNet data set. What they have are basically images like this. Natural images, photos, that kind of stuff that you can find on the internet. Each one has a class.
Alexandra Gómez Villa 1 hour 14 seconds
that they provided by humans. So the whole challenge at that time basically is composed of 1 million images and you need to predict 1000 classes. So you train with 1 million images and you should predict given an image like the class of that image. Either is a desktop computer, a flying pine and so forth and so on.
Alexandra Gómez Villa 1 hour 37 seconds
So, before...
Alexandra Gómez Villa 1 hour 39 seconds
2012, we have these results right here. This is pretty deep learning era. And the accuracy, if you see from one year from another, this gap right here is not so huge. So there was not a lot of improvement between one year and another. But
Alexandra Gómez Villa 1 hour 1 minute
Here in 2012, deep learning appeared. So this is a huge gap.
Alexandra Gómez Villa 1 hour 1 minute 9 seconds
And people were really impressive about why these guys were doing. So basically, this was using deep learning architectures to predict what are the features that better solve my problems. But you are going to
Alexandra Gómez Villa 1 hour 1 minute 28 seconds
I am going to give a really small introduction in the last block, but you are going to see into detail if you take the deep learning course. And after that, all of these architectures that always win the competition are always deep learning architectures, always.
Alexandra Gómez Villa 1 hour 1 minute 46 seconds
And more importantly, in 2015, we already surpaced the human error rate. So in this specific data set of image classification, models are already better than humans classifying between all the images. Again, in this specific sample.
Alexandra Gómez Villa 1 hour 2 minutes 6 seconds
Okay, that's a little motivation. So why neural networks matter? Because all the learning architectures are neural networks.
Alexandra Gómez Villa 1 hour 2 minutes 15 seconds
So then this is the final slide. This is just for you to go and read later. Basically, some practical advices. This is just like a summary what I say below, how to pick the number of layers or neurons, some rule up to begin with, which optimize search or
Alexandra Gómez Villa 1 hour 2 minutes 34 seconds
Or, you can see I put here and always begin with this. A regularization, you will use mostly L2, which is the default always there. And then, for hyper parameter for hyper parameter selection, as I always tell you, the best way to use cross validation if you if the resources or problem allow it.
Alexandra Gómez Villa 1 hour 2 minutes 55 seconds
But at the end, you always need to split your data between train and validation or cross validation. You never, never, never check, choose your repair parameter using your training data. Never. That's always a mistake.
Alexandra Gómez Villa 1 hour 3 minutes 15 seconds
Okay guys, any question?

Celica Krigul
1 hour 3 minutes 25 seconds1:03:25
Celica Krigul 1 hour 3 minutes 25 seconds
What would be the first hole to consider a network a deep one?
AV
Alexandra Gómez Villa
1 hour 3 minutes 31 seconds1:03:31
Alexandra Gómez Villa 1 hour 3 minutes 31 seconds
What will be sorry, what?

Celica Krigul
1 hour 3 minutes 33 seconds1:03:33
Celica Krigul 1 hour 3 minutes 33 seconds
The threshold, if you said that the network is deep, then how many layers is deep?
AV
Alexandra Gómez Villa
1 hour 3 minutes 37 seconds1:03:37
Alexandra Gómez Villa 1 hour 3 minutes 37 seconds
Yes.
Alexandra Gómez Villa 1 hour 3 minutes 40 seconds
So usually more than two layers is considered deep, but really what makes the work in deep learning is more than four layers or so. But let's say conceptually anything more than two layers is already considered deep.
Alexandra Gómez Villa 1 hour 3 minutes 59 seconds
And we have, you have shadow architectures, which are two layers, deep architectures more than two, and very deep architectures, which are deep learning architectures, which usually has, for instance, Alex Net has, if I run 15 layers, but then you have ResNet, which is from 2016, which is the neural network, which has
Alexandra Gómez Villa 1 hour 4 minutes 23 seconds
It can't even have 100 layers of of of that.
Alexandra Gómez Villa 1 hour 4 minutes 34 seconds
Okay, any other question, guys?
Alexandra Gómez Villa 1 hour 4 minutes 40 seconds
OK, so today there is no there is no practice for this part. Please, if you still have not take a look of the first of the first one, go and and check it. All of these topics are already there, the optimizer, the regularization each other, so it is nice if you take a look at that one.
Alexandra Gómez Villa 1 hour 5 minutes
Today we finished second block. I am going Monday afternoon, I am going to publish the next assignment for this block. It will be quite similar to the first one, but now we are going to focus in a completely supervised problem, right? And I will ask you to try
Alexandra Gómez Villa 1 hour 5 minutes 20 seconds
Support vector machines, neural, all of these classification methods, supervised classification methods. I will also publish, and I think.
Alexandra Gómez Villa 1 hour 5 minutes 33 seconds
Tuesday, yeah, Tuesday, maximum Tuesday afternoon, the grades for your for your assignment, and yeah, that will be all I think. Next block, we are going to, we are going to see time series, a machine learning for time series analysis, which is
Alexandra Gómez Villa 1 hour 5 minutes 51 seconds
quite interesting for you guys who work in medical data. It's pretty common. It's a pretty common data format.
Alexandra Gómez Villa 1 hour 5 minutes 59 seconds
Okay, so then let's leave it right here. Have a good weekend.