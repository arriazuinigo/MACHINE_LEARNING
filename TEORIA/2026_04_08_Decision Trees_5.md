Alexandra Gómez Villa
0 minutes 3 seconds0:03
Alexandra Gómez Villa 0 minutes 3 seconds
OK, so.
Alexandra Gómez Villa 0 minutes 7 seconds
The screen. Make sure the screen.
Alexandra Gómez Villa 0 minutes 14 seconds
OK.
Alexandra Gómez Villa 0 minutes 18 seconds
Hey, so good afternoon. So welcome again after the after the vacation.
Alexandra Gómez Villa 0 minutes 26 seconds
I'll have a look at the of the of your assignments, but probably next week. Yeah, probably Wednesday I will. I will already review all of them, but OK, so the idea of today is to begin with another.
Alexandra Gómez Villa 0 minutes 44 seconds
Another supervised learning method. Specifically, we are going to talk today about decision trees.
Alexandra Gómez Villa 0 minutes 53 seconds
Let me minimize this.
Alexandra Gómez Villa 1 minute 5 seconds
OK. So this is a little bit of the outline that we are going to follow, right summary basically really we are going to go really deep today even into decision trees and the basic stuff that you can at least.
Alexandra Gómez Villa 1 minute 22 seconds
later go and use a machine learning library like Scikit-learn as we always do. And at the end we are going to be, we're going to look at really, really quick sample of random forest which is basically a more a combination of decision trees and it's a really, really
Alexandra Gómez Villa 1 minute 41 seconds
Powerful algorithm. I say liberal introduction because at the end Random Forest is going to be. If you look at the at the scale of the course, the syllabus, you're going to see that this is a topic on its own and since this has an A lecture dedicated to Random Forest.
Alexandra Gómez Villa 44 minutes 19 seconds
The same value that we are that I am trying to predict. I know that I am doing the correlate split because all the samples that are here that have a really really similar mean value all are really close to the mean and same here.
Alexandra Gómez Villa 44 minutes 38 seconds
OK, so that's the basic of decision three. Most important thing is the criteria for splitting, which is basically selecting an attribute. We try to find we try and we what we we are doing for that is trying to increment the purity and for that there are several metrics.
Alexandra Gómez Villa 44 minutes 58 seconds
See the information gain, Genius score and so forth and so on. And then we choose some criteria for stopping, which is usually the depth of the of the tree.
Alexandra Gómez Villa 45 minutes 8 seconds
There is some pruning techniques where which at this moment we we will let the the library focus into into how to prune that that tree. Pruning means make the makes the the tree less less less deep.
Alexandra Gómez Villa 45 minutes 25 seconds
And also you can also do less white, but we are going to we are letting the library manage that and then we have the random forest. So a random forest as the name says it's a combination of trees.
Alexandra Gómez Villa 45 minutes 42 seconds
Random forest is something part is is an an algorithm which is is part of the ensemble classifiers or regressors. And basically it's like this. Imagine you need to take a decision about a sample.
Alexandra Gómez Villa 45 minutes 59 seconds
So what you can do is train many models, many decision trees. So I train one. I train one decision tree with one set, one split of the data, another with another split of the data, another with another set, and so forth and so on, or perhaps with different parameters.
Alexandra Gómez Villa 46 minutes 18 seconds
The point is that each one of these models is different, this a different expert. So they are experts in something, either because they use different data, either because they use different parameters, and then at the end I need to mix the decision of all my experts.
Alexandra Gómez Villa 46 minutes 37 seconds
So for for that, usually I can use a majority vote. For instance, each one of these models takes a decision of which class is the sample. This says one, one, one and this say two. So I am taking a majority vote. So I say OK as as most of my models say one.
Alexandra Gómez Villa 46 minutes 57 seconds
This sample is class one. Or I can also have another small model here, perhaps a linear regression which basically makes the the information of all the all the all the experts. A random forest is not nothing more like this. I train several trees.
Alexandra Gómez Villa 47 minutes 15 seconds
With different sets of data and then I mix the decision of the trees, how this is, how everything is better, what we are minimizing. This is what we are going to see in the in the specific class of random forest, but as a as a way to showing you.
Alexandra Gómez Villa 47 minutes 33 seconds
How good decision trees are and more specifically random forest because random forest and this kind of ensemble methods and mixture of expert methods are the most successful usually models for data competitions.
Alexandra Gómez Villa 47 minutes 51 seconds
We have here, yeah, yeah, here I have slide like usually you you can reduce the number, the number of features in in in each splitting and so forth and so on. But OK, this is something that we are going to specifically see in the.
Alexandra Gómez Villa 48 minutes 9 seconds
In a random forest class.
Alexandra Gómez Villa 48 minutes 13 seconds
So how would are our decisions? So in 2007 Microsoft developed this device which is called Microsoft Kinet. The Microsoft Kinet is an edge device which basically take a video of the person watching.
Alexandra Gómez Villa 48 minutes 32 seconds
In front and it's able to predict this skeleton, right? Is the skeleton of the person that is in the video.
Alexandra Gómez Villa 48 minutes 40 seconds
And everything is done inside the device. This was a a paradigm change at that time because previously Nintendo, the Nintendo Wii have these these controls, these wide small controls and these these were these were using accelerometers in the device to predict the position right and you can.
Alexandra Gómez Villa 48 minutes 59 seconds
With that. But Microsoft did something a campaign saying that this this phrase you are the controller in the way that they are telling that with this device you don't need anything. You just you are you just stand up in front of the sensor and the kinet and you already get a complete reproduction of your skeleton.
Alexandra Gómez Villa 49 minutes 17 seconds
And with this, as you know, you can do dancing games, sport games, and so forth and so on, right? But how is this thing built?
Alexandra Gómez Villa 49 minutes 28 seconds
So the keynet is predicting this, which is basically a depth map. It's an inform. It's an image that's saying at each piece pixel the distance of the surface to the sensor. So darker, darker pixel values are closer to the sensor, wider are farther.
Alexandra Gómez Villa 49 minutes 48 seconds
And giving these these information of the sensor, the kinet is predicting these points right here, which corresponds to the skeleton.
Alexandra Gómez Villa 50 minutes
And basically what they did is collect a data set in which they have people in front of the sensor and then they annotate these regions. A person, well multiple persons manually annotate the color of each one of this part in the death image.
Alexandra Gómez Villa 50 minutes 18 seconds
And with that they basically have a supervised data set, right? An input.
Alexandra Gómez Villa 50 minutes 26 seconds
Sample and the output that ground through is the color of the joints. The joint what is here. So you have input data ground through output data. Now you can train a machine learning model.
Alexandra Gómez Villa 50 minutes 41 seconds
And specifically they use a random forest of three trees, so three decision trees of death 20.
Alexandra Gómez Villa 50 minutes 53 seconds
Eh.
Alexandra Gómez Villa 50 minutes 55 seconds
Were used to build the machine learning algorithm that is running inside the Microsoft kinet.
Alexandra Gómez Villa 51 minutes 3 seconds
So they use three decision trees which were trained in 1,000,000 images. This is the size of the data set. At that point they train it. They use a really high efficient implementation and they were they are they were able to train that in 1000 core cluster.
Alexandra Gómez Villa 51 minutes 20 seconds
These runs at 200 frames per second in the Kinet, right? So Kinet runs at 200 frames per second and it's able to perform these supers by this machine learning decision from here to the skeleton of the of the people.
Alexandra Gómez Villa 51 minutes 36 seconds
And that's the power of decision trees are more specifically that's the power of random forest. So one of the biggest advancements in video game industry and this is still use it a lot today in in in not in key net because it's already already.
Alexandra Gómez Villa 51 minutes 55 seconds
Already changed. We already changes paradigms, but you know they're kind of a stereo commonness. This was done with random forest with decision trees at the core. So yeah, as you can see, it's a really, really powerful algorithm.
Alexandra Gómez Villa 52 minutes 10 seconds
OK, so that's all for today. We have as always.
Alexandra Gómez Villa 52 minutes 19 seconds
I Python notebook in which you can practice a basic of of of decision trees using scikit-learn. It's really short that always it's not it's not so long, so at least you can take a look think how to solve it and ask me if you have questions.
Alexandra Gómez Villa 52 minutes 38 seconds
Um, any question?
Alexandra Gómez Villa 52 minutes 49 seconds
It doesn't seem so.
Alexandra Gómez Villa 52 minutes 52 seconds
OK. So if there is not any question, let's have a wonderful afternoon and see you on Friday guys.