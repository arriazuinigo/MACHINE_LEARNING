Alexandra Gómez Villa
0 30:03
Alexandra Gómez Villa 0 minutes 3 seconds
Can.
Alexandra Gómez Villa 0 minutes 7 seconds
OK, then if we call this.
Alexandra Gómez Villa 0 minutes 13 seconds
All right.
Alexandra Gómez Villa 0 minutes 15 seconds
OK, guys, so good afternoon. So, first thing I already post the grades of the of the assignment.
Alexandra Gómez Villa 0 minutes 22 seconds
In general, it was it was really good, of course.
Alexandra Gómez Villa 0 minutes 27 seconds
I mean, it's really difficult to get a bad grade in on assignment because you have enough time to check that everything is working, to perhaps work with your classmates and with an LLM to debug your code. I will say perhaps my only complaint, and I didn't put a lot of attention into that, is that some of you
Alexandra Gómez Villa 0 minutes 51 seconds
put a lot of text, like really, really a lot of text, pages and pages and pages. I mean, it's not necessary. More text doesn't mean like better quality of jobs. So please be careful in the next assignment not writing such amount of text because it's harder for me, of course, and it's really difficult to get into details.
Alexandra Gómez Villa 1 minute 16 seconds
Part of the job is to actually say in the report the most important observations, right? Not just saying everything. That's probably part of the human job.
Alexandra Gómez Villa 1 minute 30 seconds
OK, so, OK, so today we are we are talking about a markup chains, and now we are going to completely turn around to other to to other type of, wait, I think my screen is not anymore sharing, yeah.
Alexandra Gómez Villa 2 minutes 2 seconds
Okay.
Alexandra Gómez Villa 38 minutes 27 seconds
This stage means that in practice, you need to know some labels for how the patient is. So a doctor, a nurse, a researcher is going to tell you if the patient is, if the heart rate is in this range, is stable. If the heart rate is in other range, is critical.
Alexandra Gómez Villa 38 minutes 46 seconds
But...
Alexandra Gómez Villa 38 minutes 48 seconds
Actually, that's not quite real. That's not how real work works. For instance, for knowing if a patient is critical, the heart rate, it probably is not enough. There are other kind of variables, perhaps is the concentration of oxygen in the blood, perhaps.
Alexandra Gómez Villa 39 minutes 9 seconds
is a hormone in the blood. So you have many possible variables which tells you if you are in a state or not. So.
Alexandra Gómez Villa 39 minutes 21 seconds
You mostly depend on measurements on lab values on details. You don't need, you do not depend in a disease state.
Alexandra Gómez Villa 39 minutes 33 seconds
So.
Alexandra Gómez Villa 39 minutes 35 seconds
Since it's difficult to define what the states are, what's the state of a model is, Markov chains are quite limited in this sense, because you need to explicitly define these states, and it happens that it's not an easy job, or most of the time it's not.
Alexandra Gómez Villa 39 minutes 54 seconds
Possible to define to define completely a state given a set of observation of variables.
Alexandra Gómez Villa 40 minutes 2 seconds
This Markov chain property actually is quite powerful, as we will see, but the condition of that you have to define a specific state given in a set of variables, it's quite strict. We would like to have something more flexible that allow us to have some uncertainty.
Alexandra Gómez Villa 40 minutes 21 seconds
because perhaps I am not specifically in some defined state, they can be in part in one state, in part in another, which basically is what we are going to do next week, right? Next week, yeah, because Friday is holiday.
Alexandra Gómez Villa 40 minutes 38 seconds
Um...
Alexandra Gómez Villa 40 minutes 40 seconds
Sup.
Alexandra Gómez Villa 40 minutes 42 seconds
The take home message is basically is, yeah, Markov change is a way to model sequence systems using a discrete set of the states. We need to define this set of the space previously, and we will we also need a transition matrix, which is basically when you tell me the dynamics of what the probability of changing to one state.
Alexandra Gómez Villa 41 minutes 3 seconds
Say to the other.
Alexandra Gómez Villa 41 minutes 8 seconds
Here, just a just a clarification: RM and Markov change and complementary. These are different things, right? RMA can predict values of the signal. So, for instance, in our case, it will predict heart rate, but Markov change predict.
Alexandra Gómez Villa 41 minutes 25 seconds
AE state classes.
Alexandra Gómez Villa 41 minutes 33 seconds
In our next class, what we are going to say is basically, okay, it's difficult to define a state, to completely define what the state is. So what we are going to do is say, let's say that the states are not observable,
Alexandra Gómez Villa 41 minutes 50 seconds
Are hidden.
Alexandra Gómez Villa 41 minutes 52 seconds
hidden from us. And that's how we are going to change from Markov change to hidden Markov models. Basically, it's a generalization of Markov change following this reasoning that actually is not possible to define states most of the time. So let's assume that we cannot observe at the state.
Alexandra Gómez Villa 42 minutes 15 seconds
and we are going to use the dynamics of Markov chain, the transition metric to guess to get observable states that is going to tell you again if we are this in our example, if we are healthy, we are deceased or not.
Alexandra Gómez Villa 42 minutes 33 seconds
Okay.
Alexandra Gómez Villa 42 minutes 36 seconds
That will be the class of today, guys, which...
Alexandra Gómez Villa 42 minutes 40 seconds
I think it's a good thing that was quite short because we have some technical issues. Sorry about that. I'm going to check what happened. So today, I didn't put any Python notebook because I prefer that next class you see you mostly work in hide and mark of models.
Alexandra Gómez Villa 43 minutes
Markup change for us is mostly a base, a tool for learner hybrid Markov model, so I don't see a point in having a notebook about markup change, really.
Alexandra Gómez Villa 43 minutes 14 seconds
So any questions?
RM

Raians Sprogis Marons
43 minutes 18 seconds43:18
Raians Sprogis Marons 43 minutes 18 seconds
I don't have a question, but more of a sort of notice. I wrote it down in the like questions forum for the class, but the previous recording from Friday didn't seem to upload to Virtual Campus and also it's not available to access through Teams.
AV
Alexandra Gómez Villa
43 minutes 22 seconds43:22
Alexandra Gómez Villa 43 minutes 22 seconds
Yes.
RM

Raians Sprogis Marons
43 minutes 39 seconds43:39
Raians Sprogis Marons 43 minutes 39 seconds
Like you need to have access granted for some reason, so just wanted to call attention to that.
AV
Alexandra Gómez Villa
43 minutes 44 seconds43:44
Alexandra Gómez Villa 43 minutes 44 seconds
Um...
Alexandra Gómez Villa 43 minutes 46 seconds
Yeah, someone wrote me about it.
Alexandra Gómez Villa 43 minutes 50 seconds
So, you cannot see the video in the app I am sharing a screen, so we can check.
Alexandra Gómez Villa 44 minutes
So let me just be sure about Dat.
Alexandra Gómez Villa 44 minutes 4 seconds
So, if you go here...
Alexandra Gómez Villa 44 minutes 15 seconds
Um...
Alexandra Gómez Villa 44 minutes 18 seconds
Here, for instance, see details.
Alexandra Gómez Villa 44 minutes 24 seconds
This video.
Alexandra Gómez Villa 44 minutes 27 seconds
If you click here, you cannot see this via.
RM
Raians Sprogis Marons
44 minutes 31 seconds44:31
Raians Sprogis Marons 44 minutes 31 seconds
Correct. Yeah, it just says that you can't access it.
AV
Alexandra Gómez Villa
44 minutes 35 seconds44:35
Alexandra Gómez Villa 44 minutes 35 seconds
Okay, but you can, but you see that the video is here, right?
RM
Raians Sprogis Marons
44 minutes 36 seconds44:36
Raians Sprogis Marons 44 minutes 36 seconds
I requested access, but I don't think that goes to you. Pardon me?
AV
Alexandra Gómez Villa
44 minutes 40 seconds44:40
Alexandra Gómez Villa 44 minutes 40 seconds
Okay.
Alexandra Gómez Villa 44 minutes 44 seconds
Yeah, actually.
Alexandra Gómez Villa 44 minutes 46 seconds
The the the access is not is not like I don't have any notification or something, as long as I can I can give you access, so I need to write down to to I need to write an email to IT because yeah, I I am not the one who can actually give you the access, which is quite a strange because most of the most of the previous ones.
Alexandra Gómez Villa 45 minutes 10 seconds
You could see it right here. It's right. You are right. I also saw in the in the platform that is not uploaded in the platform, but I cannot directly load that to CAMPUS VIRTUAL either. I don't have the administrative privileges to do that. So
Alexandra Gómez Villa 45 minutes 29 seconds
Um...
Alexandra Gómez Villa 45 minutes 31 seconds
What I can do mostly is write an email to IT and ask them to check what happened with the course. But please also write you the email, because it's better if you also write and and say, hey, I cannot access this video in the Teams.
Alexandra Gómez Villa 45 minutes 52 seconds
of the course, what's happening, what I don't have permission.
RM
Raians Sprogis Marons
45 minutes 56 seconds45:56
Raians Sprogis Marons 45 minutes 56 seconds
Okay, yeah. Is there a specific email that I should be reaching out to then?
AV
Alexandra Gómez Villa
45 minutes 58 seconds45:58
Alexandra Gómez Villa 45 minutes 58 seconds
What's up?
Alexandra Gómez Villa 46 minutes 3 seconds
I have an email, but I don't know if it's
Alexandra Gómez Villa 46 minutes 9 seconds
Because.
Alexandra Gómez Villa 46 minutes 13 seconds
Let me check now, let me check later, because I indeed I have an email, but it's perhaps for more administrative stuff for teachers and things like that.
Alexandra Gómez Villa 46 minutes 24 seconds
Yeah, I will need to check the page and let you know. If not, you can also, you can always check because here in the platform, we don't have that either.
RM
Raians Sprogis Marons
46 minutes 32 seconds46:32
Raians Sprogis Marons 46 minutes 32 seconds
Mhm.
Raians Sprogis Marons 46 minutes 36 seconds
Okay, that makes sense.
AV
Alexandra Gómez Villa
46 minutes 37 seconds46:37
Alexandra Gómez Villa 46 minutes 37 seconds
Yeah, yeah, yeah, no, let me check. Really, I don't know the email. I need to check. I need to to check if perhaps write an email to the to the director of the master and he told me the email of the IT guys.
RM
Raians Sprogis Marons
46 minutes 54 seconds46:54
Raians Sprogis Marons 46 minutes 54 seconds
Okay. Thank you.
AV
Alexandra Gómez Villa
46 minutes 57 seconds46:57
Alexandra Gómez Villa 46 minutes 57 seconds
Okay, you're welcome. Also, check if you have, well, I will check later if it's if this video is uploaded to CAMPUS VIRTUAL. For some reason, all the videos were uploaded into CAMPUS VIRTUAL and from last Friday, from last Wednesday, they suddenly stopped uploading to the platform.
Alexandra Gómez Villa 47 minutes 15 seconds
But OK.
Alexandra Gómez Villa 47 minutes 19 seconds
Okay, so... AE.
Alexandra Gómez Villa 47 minutes 23 seconds
Yeah. Any other question?
Alexandra Gómez Villa 47 minutes 28 seconds
OK, so then have a nice rest of afternoon, and I will check that that now that.
Alexandra Gómez Villa 47 minutes 36 seconds
OK, have a nice afternoon, guys.