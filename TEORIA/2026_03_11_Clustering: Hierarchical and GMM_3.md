Alexandra Gómez Villa
0 minutes 25 seconds0:25
Alexandra Gómez Villa 0 minutes 25 seconds
OK.
Alexandra Gómez Villa 0 minutes 27 seconds
OK, so.
Alexandra Gómez Villa 0 minutes 31 seconds
So basically the fun today is finish our basics and unsupervised learning block, our first of these four blocks of the of the the machine learning course.
Alexandra Gómez Villa 0 minutes 44 seconds
So the idea to lay basically is to.
Alexandra Gómez Villa 0 minutes 48 seconds
And our our hierarchical clustering part, basically we didn't see anything about hierarchical clustering. So we are going to take a look about basic and hierarchical clustering strategies, motivation etcetera and then we are going to.
Alexandra Gómez Villa 1 minute 7 seconds
Focus the rest of the of the class, mostly in Gaussian mixture models, which is under kind of unsupervised learning strategy, which of course can be used for clustering. Finally we will dedicate a couple of.
Alexandra Gómez Villa 1 minute 25 seconds
Well, Thomas, like just summarizing everything about about clustering regarding which method should you select depending on the topic, which metrics should you use, etcetera, etcetera.
Alexandra Gómez Villa 47 minutes 26 seconds
Just more general advice. It will be trying to select that clustering algorithm is in practice you need to experiment with many of them. So multiple of course is the most straightforward thing is to begin with with Kmean, the most simple of all of them.
Alexandra Gómez Villa 47 minutes 45 seconds
And then we check the performance of all of them with metrics like silhouette, visual inspection using of course visualization, TSNE, PCA.
Alexandra Gómez Villa 47 minutes 58 seconds
And then you compute metrics, metrics in general you have here. So we have internal metrics as in which you don't need labels, for instance the silhouette score that we discussed about. But of course there are other metrics they be folding doing index to compounding in cycle earn.
Alexandra Gómez Villa 48 minutes 16 seconds
Then you have external metrics, the ones that need labels. Then we have here the area and then mood normalizing mood poll information.
Alexandra Gómez Villa 48 minutes 26 seconds
Visual inspection. Again, this is really useful if you have labels or if you can visualize the data, this is always useful to do. Then you have here PCA and we have order visualization techniques, DSME, UMAT. Then for the structure of the data set for finding the number of K or for looking how.
Alexandra Gómez Villa 48 minutes 46 seconds
The data just look the Elbow method, dendograms from hierarchical um as we saw, then density plots um.
Alexandra Gómez Villa 48 minutes 55 seconds
And let's say that most.
Alexandra Gómez Villa 49 minutes 2 seconds
Practical No. Yeah, a more practical and pragmatic advice is something like this. Usually K means work really well for compound and spherical clustering. We already saw that methods like DBS scan and HDBS scan handles arbitrary checks and noise hierarchical clusterings.
Alexandra Gómez Villa 49 minutes 21 seconds
They are good for necessary structure, respective clustering, useful or high-dimensional spaces. Gaussian mixture model are also good here, for instance.
Alexandra Gómez Villa 49 minutes 31 seconds
Um.
Alexandra Gómez Villa 49 minutes 34 seconds
That will be all. Any question before I talk about the assignment?
Alexandra Gómez Villa 49 minutes 46 seconds
No.
Alexandra Gómez Villa 49 minutes 50 seconds
OK, now it doesn't seem so. OK, so guys, so I already.
Alexandra Gómez Villa 49 minutes 56 seconds
So I already pulled the assignment in Campus Virtual. Let me show you.
Alexandra Gómez Villa 50 minutes 9 seconds
So you can find the assignment one here. As you can see, this is basically a PDF describing the the assignment.
Alexandra Gómez Villa 50 minutes 26 seconds
So the idea.
Alexandra Gómez Villa 50 minutes 29 seconds
As you see is that we are going to focus in clustering for the assignment. So the idea is to use the heart disease data set which is a really really common data set for machine learning when when anyone is learning machine learning and this data set which is a policy available here.
Alexandra Gómez Villa 50 minutes 48 seconds
Thirsting is.
Alexandra Gómez Villa 50 minutes 51 seconds
You need to use a specifically this file processed Cleveland data.
Alexandra Gómez Villa 50 minutes 58 seconds
And if you if you look at these data you will you will see that these data have these features right here. Basically it's a file and each row is is a sample and the sample have these features right here. So important thing we are not going to use the target.
Alexandra Gómez Villa 51 minutes 16 seconds
Only for feature selection for the final results, but for overall we are not going to use the target because this is unsupervised learning clustering. So you are supposed to not know which which which are the ground through of your samples.
Alexandra Gómez Villa 51 minutes 33 seconds
So then basically what I what I am asking you here is basic stuff really what we have been doing. So a loaded data set, a missilent handle, a missing values, knows that kind of stuff in which we saw in data cleaning. You plot the distributions, histograms, box plots.
Alexandra Gómez Villa 51 minutes 53 seconds
Correlation heat maps.
Alexandra Gómez Villa 51 minutes 57 seconds
Then try to do some preprocessing and feature engineers. So first you need to encode the categorical features that we saw. Then you need to normalize of the standardized features. Then you are asked to perform a dimensionality reduction to plot to see the the the data.
Alexandra Gómez Villa 52 minutes 17 seconds
Task to perform feature selection. Here of course you can use the labels for using better feature selection algorithms. Then we have task 3/4 and five.
Alexandra Gómez Villa 52 minutes 33 seconds
Well, these three tasks, basically each one of these is focused on one clustering algorithm. So K means Gaussian mixture models, hierarchical clustering and you are asked to compute this kind of the same thing that we talk about.
Alexandra Gómez Villa 52 minutes 48 seconds
The elbow method try with several number of K visualize the results kind of the same for all of them. Then at the end you need to do some kind some kind of evaluation or how they're good at the result with the algorithms. So you are asking to compute Ari here for instance.
Alexandra Gómez Villa 53 minutes 8 seconds
And to say something about the data and the results which are recovered to proposition methods. And then in addition to the code, what you need to do code is describing in each task.
Alexandra Gómez Villa 53 minutes 23 seconds
You need to do a writing report of maximum to to 2500 words and we seen these questions. So one important thing here so.
Alexandra Gómez Villa 53 minutes 36 seconds
Of course, in machine learning it's important to do the program part, the programming part. So that's why I'm asking for coding. I am asking you to give me a Jupiter notebook. I want to check the results, everything. But machine learning even more than the than the methods or the coding part is really, really.
Alexandra Gómez Villa 53 minutes 55 seconds
Important the analysis that you do about the models. So the report at the end is trying to say which model is better, what model is behaving better with the data, what I am observing in the data. There seems to be two groups, there seems to be people have.
Alexandra Gómez Villa 54 minutes 15 seconds
These features have more probabilities of having the illness, et cetera, et cetera. So the writing report is really important to check that you are using machine learning to analyze data, right? Not just please just not consider that the most important part is the coding.
Alexandra Gómez Villa 54 minutes 34 seconds
Coding is important, but even more important is the conclusions that you get with your coding, right? So please the idea you you deliver the Jupiter notebook with the code. Please comment the code and whatever you can report in PDF.
Alexandra Gómez Villa 54 minutes 52 seconds
In a single seat here I put a a a rubric kind of a general lines that one will be evaluated and this is some kind of advice. So as you are four or five people per group, is that good idea to divide the pipeline?
Alexandra Gómez Villa 55 minutes 11 seconds
Of the of this of this work depending on how you how you would like to work. Of course each group organize itself, but a good idea is to divide task one member per task and then you need to do a join review session for for before the submission.
Alexandra Gómez Villa 55 minutes 28 seconds
And task six and seven of course need to be need to be done with all them, all the team members. The deadline for this is.
Alexandra Gómez Villa 55 minutes 39 seconds
30 of March here in the platform of course. If you have questions of course you can you can send me an e-mail. I expect you in general to be able to manage coding coding errors so.
Alexandra Gómez Villa 55 minutes 56 seconds
I don't expect you to ask me question about this is not compiling Alexandra or so coding questions. I am not expecting you to ask me coding question because this is not a coding course, but machine learning questions are OK. Anything about the data, anything about the models, anything about the task per se, it's OK to ask me.
Alexandra Gómez Villa 56 minutes 15 seconds
Either posting the forum, either send me an e-mail as as always you do, it's OK.
Alexandra Gómez Villa 56 minutes 21 seconds
Today, so today there is no there is no practical session for these Gaussian mixture models or hierarchical cluster because it's better if you begin to read the assignment, use this time to understand why I'm asking for how are you, how are you going to split the word, etcetera, etcetera.
Alexandra Gómez Villa 56 minutes 41 seconds
So use the time for that better.
Alexandra Gómez Villa 56 minutes 46 seconds
Uh, any question regarding the assignment?
Alexandra Gómez Villa 56 minutes 55 seconds
OK.
Alexandra Gómez Villa 56 minutes 57 seconds
So if there is no more questions, then have a good day and see you on Friday then.