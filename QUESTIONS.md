# Questions
Please try to answer the following questions.
Reference your own code/Notebooks where relevant.

## Your solution
1. What machine learning techniques did you use for your solution? Why?

   *An adaptation of nested-cross validation for time series.
   Time series differ from the usual data due to its dependency on the time information thereby the usual train-test split is not directly applied. I could have done a hold-out cross-validation but that would mean that I would only be evaluating the model in one subset of the data. With this in mind, I performed several splits of the data, going from 50% to 90% with increments of 10% each time to verify how the error generalize.*

   *As I didn't performed no hyperparamenter tuning, I didn't performed the 'inner' loop of the nested-cross validation*

   *The model used to predict the electrical usage is an additive regression model with a linear curve trend component, weekly and daily seasonality components.*

   *From the exploratory analysis we observed that we would have a daily and a weekly seasonality and therefore it felt like a good shoot for a preliminary analysis.*

   *The model was fitted using the prophet library from Facebook
   (https://research.fb.com/prophet-forecasting-at-scale/) with the default settings.*

   *The prophet model was developed for usage on business common time series forecasting problems so might not be the most optimize solution for this problem but I believe that will provide a good idea of the error and serve as comparison point for further models.*

   *Given the properties of the data, I could had tried also a LSTM.*

2. What is the error of your prediction?

 	*The error is measured with Normalized Root Mean Squared Error (NRMSE) (using the range of values in the dataset as the normalization factor), so we can compare between houses and we achieved 15.5% on average for the tests sets.*

   a. How did you estimate the error?

   	*I used the NRMSE pairing with nested cross validation as explained above on the different test sets. We predicted the last week of each training set, to have an estimate for training, and used the week ahead as testing set.*
   
   b. How do the train and test errors compare?

      *They are of similar magnitude.*
   
   c. How will this error change for predicting a further week/month/year into the future?

   *It will definitely increase... the model does not account with yearly seasonality, and the further we go in the future (away from the data that was seen during the training period) the uncertainty around the different components increases and therefore the error will increase accordingly.*
   
3. What improvements to your approach would you pursue next? Why?

	*I would try to include the calendar holidays that are known and the school holidays. On those days there is a change on the routine leading to an increase/decrease on the consumption that would be expected for that day of the week.*

	*I would like to explore a way of include the household 'holidays' (very low consumption periods) information, maybe try to solve it as a classification problem that would be then feed as the normal calendar holidays for the prediction. Again, I believe that those are not being capture by the additive components of the model as it is now. Trend would had to adjust to those particular events in order for the forecast to increase or decrease on consumption and compensate the 'break' on the daily and weekly seasonality.*

	*Explore an LSTM as a solution for this problem, include temperature info, include holidays info*

4. Will your approach work for a new household with little/no half-hourly data? 

	*Not really...*

   How would you approach forecasting for a new household?

   *Probably would collect as most descriptive data from the household as possible, and try to match it with the closest one in terms of those descriptions and use the model of that house for the initial predictions of the new household. Once I had some weeks of data I would try to find the closest household in terms of weekly pattern with the new one and use that model instead.*
