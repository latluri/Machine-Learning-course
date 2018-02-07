This project is a combined effort of Laxmi Narayana, Marcia, Punniya darshan and Chanakya
This project is a part of Data Sciences for social good (DSSG), an initiative by the Data sciences department of UNC Charlotte
Aim of the project is to analyse and solve a social challenge with the help of analytics


We build a classifiers to predict whether a school was in the "A/B" or "C/D/F" grade i.e two classes “good” and “poor” performing schools.
We used all schools in NC (about 2200) in our analysis and to train our model (we mixed elem, middle, high schools altogether). We build multiple models with inclusion and exclusion of features such as 
Poverty Indicator :Whether or not to use the poverty indicator to predict the school performance
Diversity Indicator :Representing diversity of the students
Teachers' experience
Type of funds etc

When we include a poverty indicator in the set of features (variables) used for prediction, our accuracy was 81%.
When we excluded any direct poverty indicator (such as % free/reduced cost lunch or % economically disadvantaged students) but used other features like teacher performance, # of suspensions, racial diversity, etc, we were still able to predict a good versus a poor performing school with 70% accuracy.

Our bottomline takeaway is that we believe our results suggest that CMS may need to redistribute more than just SES disadvantaged students, as teacher performance and experience, suspensions, and diversity in addition to poverty are contributing factors in our model to school performance grade.
