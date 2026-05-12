Alexandra Gómez Villa
0 minutes 3 seconds0:03
Alexandra Gómez Villa 0 minutes 3 seconds
It seems that it's working. Let me know if you, if you don't see that the recording is happening, please.
Alexandra Gómez Villa 0 minutes 11 seconds
Okay, so...
Alexandra Gómez Villa 0 minutes 14 seconds
So, good afternoon. The idea today is to begin with a work.
Alexandra Gómez Villa 0 minutes 19 seconds
which basically time series analysis. So until now, we were mostly focused on signal or samples, which are basically independent among them. Basically, we have samples, like it doesn't matter the order or it doesn't matter any relationship between them.
Alexandra Gómez Villa 0 minutes 40 seconds
and we consider each one an independent sample. This is part of the idea assumption and this is quite useful when you are trying to deal with a static data like images or perhaps like tables. But
Alexandra Gómez Villa 1 minute
As you probably know, there are many, many signals that depends on time or some kind of sequence characteristic that we can model and we can use. And actually the sequence nature of these kind of signals is really, really important to try to model them, to try to infer
Alexandra Gómez Villa 1 minute 21 seconds
And take Dat.
Alexandra Gómez Villa 1 minute 24 seconds
OK, so let's, let's, and I can show you some examples. So, the idea, yeah, first is a quick intro in what our time series, what we are interested in that, what what's difference our from previous a classic machine learning that we were doing, then we are going to talk about this.
Alexandra Gómez Villa 1 minute 44 seconds
Three, really important.
Alexandra Gómez Villa 1 minute 48 seconds
concepts in time in classic, time C, which is a stationarity, autocorrelation and windowing. And these basically will allow us to build some basic knowledge about how to treat this data and
Alexandra Gómez Villa 52 minutes 12 seconds
This, so now you have, yeah, and you continue like that, and you predict the next one, the next one, and so forth, and so on, one after the other, like this.

José Pablo Soriano Torres
52 minutes 13 seconds52:13
José Pablo Soriano Torres 52 minutes 13 seconds
Okay.
AV
Alexandra Gómez Villa
52 minutes 24 seconds52:24
Alexandra Gómez Villa 52 minutes 24 seconds
Always using the previous data after you move your window and the new data that you that you that you predicted you add that to the signal. That's why how that's how you predict.

José Pablo Soriano Torres
52 minutes 36 seconds52:36
José Pablo Soriano Torres 52 minutes 36 seconds
So, in that case, you need a stride of 1 to do that.
AV
Alexandra Gómez Villa
52 minutes 41 seconds52:41
Alexandra Gómez Villa 52 minutes 41 seconds
Yeah, in this so.
Alexandra Gómez Villa 52 minutes 46 seconds
In inference time, after you already trained your model, the model is always going to give you the next value. So the model is not understanding the step. You give some window and it will give you, okay, the next value is this one.
Alexandra Gómez Villa 53 minutes 5 seconds
The next value is this one. Now, the stripe that you choose when you are making your data set matters for this. If your stripe perhaps is too high, then the model perhaps could have difficulties predicting these single values each time.
Alexandra Gómez Villa 53 minutes 28 seconds
But in general, I will tell you, it doesn't matter too much if your model is strong enough. So you use a stride, for instance, a stride which is half of the window size. But when you are doing your data set, you always predict just the next value.
Alexandra Gómez Villa 53 minutes 47 seconds
Then.
Alexandra Gómez Villa 53 minutes 50 seconds
Perhaps that will work if your if your data is simple. If not, then you need to have a really reduce this right and check again.
Alexandra Gómez Villa 54 minutes 1 second
Méan.
Alexandra Gómez Villa 54 minutes 4 seconds
Just to, just to be clear, this thing.
Alexandra Gómez Villa 54 minutes 9 seconds
Of accumulated multiple windows, one after another. This is, in theory, the procedure.
Alexandra Gómez Villa 54 minutes 17 seconds
But if you accumulate multiple windows on trying to do this, in practice, this is not so good. This doesn't work so good.
Alexandra Gómez Villa 54 minutes 28 seconds
And, but yes, to answer your question, you usually try to predict using the predefined stride. If it is not predicting, you use a smaller stride.
Alexandra Gómez Villa 54 minutes 41 seconds
And probably that that could improve, yes.
Alexandra Gómez Villa 54 minutes 45 seconds
Depend on your data, of course, yes.

José Pablo Soriano Torres
54 minutes 46 seconds54:46
José Pablo Soriano Torres 54 minutes 46 seconds
So it's not very good to like to have a lot of windows, something like that.
AV
Alexandra Gómez Villa
54 minutes 52 seconds54:52
Alexandra Gómez Villa 54 minutes 52 seconds
So it's not a lot of, it's not really good to try to predict values that are too far away. I mean, the thing with this is that...
Alexandra Gómez Villa 55 minutes 3 seconds
What is the information, the actual real information that we have in this sequence right here? So, the only real information is the one which is in the in the in the red window.
Alexandra Gómez Villa 55 minutes 17 seconds
So this one, I don't know, signal like this.
Alexandra Gómez Villa 55 minutes 23 seconds
When we...
Alexandra Gómez Villa 55 minutes 24 seconds
Begin to predict next value.
Alexandra Gómez Villa 55 minutes 28 seconds
We are basically using a predicted a predicted quantity, something that is not part of the real signal. So my use in practice models are good predicting values that are close to the original signal, like with lag one, lag two, lag three, lag 4, right? Like this.
Alexandra Gómez Villa 55 minutes 49 seconds
But as you try to predict next value, next value, what is happening is that you are not using this information because it's not part of the signal and you are using more information with what's produced by the same model.
Alexandra Gómez Villa 56 minutes 6 seconds
So, in practice, it's not really good the prediction what happens that happens here. These the memory of the models is limited, and the what if you are trying to model is quite complex in time, the model is not going to be able to predict.
Alexandra Gómez Villa 56 minutes 25 seconds
values that are really far in the future. A good example is stock prices.
Alexandra Gómez Villa 56 minutes 33 seconds
Stock prices is something that has been the obsession of economists, mathematicians, and as I said before, time series initially was solved to try to solve this problem. But what happened is that actually some series are highly stochastic and cannot be predicted very well.
Alexandra Gómez Villa 56 minutes 55 seconds
Stock prices is one of them. It's impossible to predict stock prices because the variables that you are trying to model, the quantities that you are trying to model cannot be measured only by the data that you have that these stock prices, these stock companies provide, or whether
Alexandra Gómez Villa 57 minutes 15 seconds
If you are a biologist, perhaps you have told, you have heard about theory of chaos and this kind of stuff and the butterfly effect.
Alexandra Gómez Villa 57 minutes 25 seconds
The thing is that with the prediction of series, really a small changes at some point can make your values change a lot in the future. So perhaps you can predict what happened in one, two, three, 4 steps. But if you go many steps in the future, your model is not
Alexandra Gómez Villa 57 minutes 45 seconds
designed to predict is not well, is not strong enough or doesn't have enough data or information to predict that values in the future.
Alexandra Gómez Villa 57 minutes 57 seconds
That's it.
Alexandra Gómez Villa 58 minutes
There are pretty powerful models which works better with this kind of, for doing this kind of stuff. So we are not going to look into that or perhaps one or two slides, but in deep learning or NLP, you probably want to learn about transformers.
Alexandra Gómez Villa 58 minutes 22 seconds
Performers.
Alexandra Gómez Villa 58 minutes 25 seconds
are the best model to predict time series right now. Transformers is the core of ChatGPT of LLMs. An LLM essentially, ChatGPT is solving this problem.
Alexandra Gómez Villa 58 minutes 42 seconds
Again, it's predicting the next word, giving the previous word that you have.
Alexandra Gómez Villa 58 minutes 50 seconds
I think I used to that like the question, sorry.

José Pablo Soriano Torres
58 minutes 52 seconds58:52
José Pablo Soriano Torres 58 minutes 52 seconds
Yes, it was, no, no, but it got understood very good, yeah. Thank you so much.
AV
Alexandra Gómez Villa
59 minutes59:00
Alexandra Gómez Villa 59 minutes
Okay, okay. You're welcome.

José Pablo Soriano Torres
59 minutes 1 second59:01
José Pablo Soriano Torres 59 minutes 1 second
Yeah.
AV
Alexandra Gómez Villa
59 minutes 4 seconds59:04
Alexandra Gómez Villa 59 minutes 4 seconds
So, so yeah, so then this is just a summary if you want to take a of later to this.
Alexandra Gómez Villa 59 minutes 11 seconds
Um...
Alexandra Gómez Villa 59 minutes 14 seconds
So yeah, totally this is the final slide and this is just a summary so that you can like have how the whole process is, right? So you have your row series, time series which is on process, then after we found if there is some, if there is a stationary or non-stationary and we correct in case they are not stationary, then we need
Alexandra Gómez Villa 59 minutes 37 seconds
To do some windowing.
Alexandra Gómez Villa 59 minutes 41 seconds
That window basically is going to convert our via our series of variance length into fixed size windows that we can actually process with our models. The parameters that we choose for that window is depend if we
Alexandra Gómez Villa 1 hour
perform some autocorrelation plots to check if there is some seasonality and some pattern, and that will help me choose that K and that describe. Then we perform some feature extraction. Usually, if they, if they, if they don't, if the data set of the domain itself give you some
Alexandra Gómez Villa 1 hour 19 seconds
features to extract is better if not always these 5-6 features that I told you about is really a good start.
Alexandra Gómez Villa 1 hour 29 seconds
Then, that is always cool. Same topic as before: model, super vector machine, neural network, and then you do or either your classification or your linear regression, which is to predict the next value, right?
Alexandra Gómez Villa 1 hour 45 seconds
And that's all, guys, and any question?
Alexandra Gómez Villa 1 hour 57 seconds
OK, so then, as always, I put the the...
Alexandra Gómez Villa 1 hour 1 minute
And...
Alexandra Gómez Villa 1 hour 1 minute 2 seconds
I put the in the in the Campus the notebook for this for this one, and perhaps is a little bit long.
Alexandra Gómez Villa 1 hour 1 minute 11 seconds
I was not planning to extend too much, but it was the case. It's a little bit long, but it's quite easy just to check how that process of windowing of computing the autocorrelation function is everything is with Pandas, NumPy. So try to take a look how the process is going.
Alexandra Gómez Villa 1 hour 1 minute 30 seconds
Your assignment, guys, I know, I know that I am delayed. Please give me till the end of the week. I will try to find out to finish this weekend. So if there is no more questions, I will say have a good rest of the day.