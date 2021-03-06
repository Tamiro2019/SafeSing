# SafeSing : Leveraging A.I. to balanced your technique

SafeSing is a [web app](http://18.232.171.164/) designed to help inexperienced singers identify problem areas in their voice and promote healthy singing. Most aspiring singers decide to teach themselves because professional vocal lessons are expensive (often more than 100$ per hour). Left unchecked, improper technique can lead to vocal damage such as vocal nodules and polyps. SafeSing supports novice singers by providing immediate and accurate feedback on the way in their vocal cords are coordinating to produce sound (i.e. their phonation modes). The SafeSing app takes a sung recording as input and classifies half second bins into one of three phonation categories: Breathy, Balanced, or Pressed. This classification is achieved by leveraging signal processing techniques (to convert audio into images - spectrograms), time series modeling (to isolate vocal cord sounds) and machine learning (using a convolutional neural network for image classification).

- Flask App (Folder)
	- Script to run app
	- Modules containing important functions

- Models (Folder)
	- Jupyter Notebooks (used to build and train models)
	- Trained Model Files
	
