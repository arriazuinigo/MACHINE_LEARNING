Alexandra Gómez Villa
0 minutes 3 seconds0:03
Alexandra Gómez Villa 0 minutes 3 seconds
Now let's continue. OK, so basically in this is in in this session and the next two sessions we are going to focus on this part of the basic machine learning pipeline.
Alexandra Gómez Villa 0 minutes 17 seconds
Um, here we have data collections and.
Alexandra Gómez Villa 0 minutes 23 seconds
I will say most of the time when you are working in a company, this is not your task. When you are working in machine learning, this is the task of the expert or the company or the client who is basically in in front of the source of the data.
Alexandra Gómez Villa 0 minutes 40 seconds
For instance, if you are, if you are a nurse, if you are a doctor, you are attending people, then those are the ones who are recollecting the data, either in a in in a computer system, either in whatever, whatever topic.
Alexandra Gómez Villa 0 minutes 55 seconds
But a.
Alexandra Gómez Villa 0 minutes 58 seconds
Most, well, most of the time when you are beginning to implement a machine learning project as you are the expert. And when I say the expert is the person who is going to do all of this stuff right here, right? You know you are going to prepare the data, you are going to select features you are going.
Alexandra Gómez Villa 1 minute 18 seconds
To do machine learning selection and trained on model. So most of the time when you are trying to gather some data, you will be at this point telling people which data is useful for you that.
Alexandra Gómez Villa 1 minute 34 seconds
It's really common, but it could happen that they already have a data set and then you begin here, OK.
Alexandra Gómez Villa 1 minute 41 seconds
So then talking about these boxes explicitly with this is the our focus today and next to the slides as I already told you.
Alexandra Gómez Villa 1 minute 50 seconds
OK, so let's begin then. Let's say that we want to build a fruit classifier. Let's say that we have some fruits. We have apples, oranges, lemons and mandarins.
Alexandra Gómez Villa 50 minutes 9 seconds
Let's do a box plot or a scatter plot. It seems to be some outliers in these classes, in these, in these other classes, not. So actually it's really, really important to have.
Alexandra Gómez Villa 50 minutes 23 seconds
Good abilities or to have have some criteria to use visualization, visualization plots and basic statistics which are part of the visualization plots.
Alexandra Gómez Villa 50 minutes 35 seconds
OK.
Alexandra Gómez Villa 50 minutes 38 seconds
So that's from the theory part. Um.
Alexandra Gómez Villa 50 minutes 42 seconds
About the notebook. So if you go to to the to the page of the course the virtual campus.
Alexandra Gómez Villa 50 minutes 53 seconds
You will see that now.
Alexandra Gómez Villa 50 minutes 56 seconds
If I am, yes.
Alexandra Gómez Villa 51 minutes 1 second
We have basically in the slides we have the slide that I just that I just presented. I will correct the the typo that I have there and we have a Python notebook session. So basically we are going to work like this, right? I present a slide, some theory will be here and the corresponding Python notebook for that slide is.
Alexandra Gómez Villa 51 minutes 21 seconds
Here.
Alexandra Gómez Villa 51 minutes 22 seconds
How to work with the Python notebooks? If I understand correctly, you already saw a Python course or Python programming, right?
Alexandra Gómez Villa 51 minutes 33 seconds
Yes, yeah, because that will be up on it now and.

Iris Mestres Pascual
51 minutes 34 seconds51:34
Iris Mestres Pascual 51 minutes 34 seconds
Yes.
51:36
51 minutes 36 seconds
Yes.
AV
Alexandra Gómez Villa
51 minutes 40 seconds51:40
Alexandra Gómez Villa 51 minutes 40 seconds
So then basically you can you can work with the Pythons in Python notebooks in two possible ways, either locally using Visual Studio or using Spider. So about that question.
Alexandra Gómez Villa 51 minutes 56 seconds
Are you familiar with Python notebooks?
Alexandra Gómez Villa 52 minutes 1 second
Yes, OK, thanks. And where where did you work with the Python notebooks in Spider, in Google Colab, in Visual Studio?

Iris Mestres Pascual
52 minutes 12 seconds52:12
Iris Mestres Pascual 52 minutes 12 seconds
VS code.
AV
Alexandra Gómez Villa
52 minutes 14 seconds52:14
Alexandra Gómez Villa 52 minutes 14 seconds
Yes, code. OK, thanks.

Ludovic Méan Touroyan
52 minutes 14 seconds52:14
Ludovic Méan Touroyan 52 minutes 14 seconds
As well.

Iris Mestres Pascual
52 minutes 16 seconds52:16
Iris Mestres Pascual 52 minutes 16 seconds
Yeah, I'll use Jupiter notebooks like.
AV
Alexandra Gómez Villa
52 minutes 17 seconds52:17
Alexandra Gómez Villa 52 minutes 17 seconds
OK.
Alexandra Gómez Villa 52 minutes 20 seconds
OK, so then.
Alexandra Gómez Villa 52 minutes 24 seconds
You can choose to work with them in in in whatever platform you feel is better for you. If you're using Visual Studio, it's as simple as we have here the the the notebook. This is the notebook that you will work today.
Alexandra Gómez Villa 52 minutes 40 seconds
I already prepare this part if you are using a Google app. If not you just need to change this part for your your your PC or and remove this.
Alexandra Gómez Villa 52 minutes 56 seconds
But and then you work with the basically with the well, I will continue with that later. But if you prefer to work with Colab, so Colab is basically online Python notebooks. So is this free is this free resource from Google.
Alexandra Gómez Villa 53 minutes 16 seconds
Basically, if you go to this webpage, Collab Research, let me share that with you in case someone does not know about it.
Alexandra Gómez Villa 53 minutes 28 seconds
So basically if you go there, you will have access to this kind of stuff with Google. Yeah, you need probably you need to put your use your Google account, but that's on you. If you don't want just use use it locally, use Visual SBS code.
Alexandra Gómez Villa 53 minutes 44 seconds
And then basically you can create new notebooks here as you see.
Alexandra Gómez Villa 53 minutes 51 seconds
And if my Internet works, yes. So this right here guys, is a Python notebook. I can do here any stuff that I want, right?
Alexandra Gómez Villa 54 minutes 9 seconds
I.
Alexandra Gómez Villa 54 minutes 12 seconds
That here is connecting.
Alexandra Gómez Villa 54 minutes 19 seconds
Then it's already there, so I can I can already do a stuff here, right?
Alexandra Gómez Villa 54 minutes 26 seconds
To whatever. It's a Python notebook. It's exactly a Python notebook. So this is pretty useful for those of you who are not familiar with this because it's it's really on the go, right? I just can go to Internet, create a new call up and then I can do what?
Alexandra Gómez Villa 54 minutes 45 seconds
Whatever basic machine learning of Python code here, but it's a nice tool, especially when you are learning. Of course it's not a tool for big projects or that kind of stuff, but when you are learning, it's really good.
Alexandra Gómez Villa 55 minutes 3 seconds
Of course there is a part that you need to learn about is about these files right here. So this is basically an instantiation of a virtual machine in a remote server in the cloud. So the data paths, the folders that we are accessing are here.
Alexandra Gómez Villa 55 minutes 20 seconds
So if you are using data, you need to put your data here. So if you check the Python notebook for those of you who want to try call up, you can check here that I'm using this part right?
Alexandra Gómez Villa 55 minutes 37 seconds
And this part is the part that should that should go here. There is many documentation about that. Just send me an e-mail or something. Maybe you want a specific pointing in how to do that. It's not hard, but if you are confused or don't know how to do it, it's OK, just tell me.
Alexandra Gómez Villa 55 minutes 53 seconds
OK, now returning to the specific to the specific notebook. So here we are going to work with fruits of course, because it's the topic of the class and we have this nice data set that you need to do download from GitHub. It's available free, freely available and you can see this kind of a stuff, right?
Alexandra Gómez Villa 56 minutes 12 seconds
Fruit label, fruit name, fruit substitute, mass with a color score. So we were we were mostly playing with these two attributes, mass and width, but there are other scores. So in order to download this, it's as easy as going here and download raw file, right?
Alexandra Gómez Villa 56 minutes 30 seconds
And with this profile you will have it in your in your computer and then once you have it in your computer you put the path here. Or if you want to run it in Google Colab, then you need to put it in Google Colab right here in this folder.
Alexandra Gómez Villa 56 minutes 47 seconds
And that's all so.
Alexandra Gómez Villa 56 minutes 51 seconds
I invite you to use this last half an hour to explore the notebook is really basic notebook. It's not it's not in nothing hard. Just check the visualization, check the conclusion, check what's there, how the how the features looks like.
Alexandra Gómez Villa 57 minutes 9 seconds
Um.
Alexandra Gómez Villa 57 minutes 11 seconds
That's all. Any question?
Alexandra Gómez Villa 57 minutes 20 seconds
Yeah, doesn't seem so. Sometimes you. OK, so if there is no questions, have a good weekend and see you next week. And for the person who asked about the quantity, I will, I will put something.
Alexandra Gómez Villa 57 minutes 40 seconds
Fountain in the in the in the virtual campus so you can understand how it's how it's computed by hand.