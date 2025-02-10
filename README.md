# Tennis_Ace_Prediction

This project takes in data of tennis serves, to produce an end-to-end data science project to predict whether a serve is an ace or not. The prediction can be accessed locally by running app.py. The model 'xgb_model.joblib' can be used to predict on new test sets (see 5.Test_model.ipynb).

In the data understanding and visualisation section, the data is explored and visualised. It can be seen that even though the dataset is only supposed to contain "valid" serves that are called in, there are many serves that land outside the service boxes. This data has to be carefully cleaned. There are also plenty of cases where the ball bounce and return are on opposite sides of the court, which is a fault according to the rules of tennis. Due to time constraints, I felt like I could not clean the data as thoroughly as I wanted. For example, regarding serves that fell close to the net, it can only be possible if the serve speed is low. Fast serves either hit the net (out or let, not counted as a serve) or go towards the middle of the court. Only slow, underhand serves can possibly reach the part of the court near the net. Height of ball_hit_z was can also be carefully examined based on how high to tennis players usually jump.

The feature enginnering section produces three new features that are listed in the top 10 most important features according to a random forrest classifier. I felt the placement of the serve in relation to where the returner was positioned would matter a lot. Also the placement of the serve close to the lines was predicted to matter. These metrics do indeed matter.

Again, due to time constraints, I cut the model training short and went ahead by using only two models. The confusion matrix has to be examined and data points that were misclassified has to be carefully looked at. 

Regarding the wider context in tennis, skill of the server and returner definitely comes to mind. A server has great skill if they are able to position balls as far away from the returner's position as possible, with high speed. A returner has great skill if such a serve is not an ace, ie the returner is able to predict where the ball would land.

