# Personal Loan Default Probabilities & Risks

Tiancheng Xu, Bryan Huang, Weiqi Li, Yashuo Wang (Gloria), Jose Zuart

Oct 2021

## Role & Main Tasks in the Project





## Business Understanding
According to information presented by the World Bank, the percentage of nonperforming loans to total gross loans in India has been rising dramatically since 2011. Currently, 7.39% of the total gross loans were nonperforming loans, which is about 200% higher than the level in 2011. Defaulting on a personal loan is generally considered as a civil dispute in India. Once an individual defaults on their loan, banks will have to hand them over to professional recovery agents. Eventually the banks will need to resort to either a “Hair Cut” or the SARFAESI Act to recovery at least part of the loan, a time- and money-consuming process.

To minimize the chance of customers defaulting on their loans, commercial banks require applicants to go through a very detailed screening process. During the process, banks want to improve their ability in identifying individuals with higher risks of defaulting on loans. Once identified, these individuals can be passed on to further scrutiny/analysis. Our motivation for the project comes mainly from the belief that more accurately estimating defaults can considerably increase the profitability of banks.

Banks or financial institutions with more powerful data mining techniques and forecasting models will have an advantage in being able to identify the probability that a client defaults, then charge rates and fees or turn down loan applications more accurately (to different customers), thus maximizing profits while lowering risks. According to Stein's research work on the relationship between default prediction and lending profits, a bank can increase its profitability by around 11 basis points (BPS) for every dollar, which translates into a possible 0.11% increase in the income for every dollar in loans. According to Standard & Poors the market size for housing loans in India is 280 billions of dollars. Therefore, there can be a significant increase in the profitability of the business.

## Data Understanding
We obtained the public personal loan data from Kaggle.com. The dataset consists of 252,000 samples from the Indian commercial banking industry, with each sample representing the individual’s personal information as well as the status of the risk flag that is given by the bank. 13 features are available in the dataset, including Id, Income, Age, Experience, Married/Single, House Ownership, Car Ownership, Profession, City, State, Years on the current job, Years on the current house and Risk flag.

Among all these variables, Income, Age, Experience, Years on the current job and Years on the current house are numeric variables that take on consecutive values, while the following are categorical variables:

Married/Single (single, married), House Ownership (rented, norent_noown, owned), Car Ownership (no, yes), Profession (51 distinct professions), City (317 cities), State (29 states). Risk Flag is the target variable that we will need to predict, and it is a binary variable that takes the value of either 0 or 1.

## Data Preparation
For the preparation of the data, a statistical summary of the variables was made, as well as checking that there was no missing data in the columns (Graph A). 

<img width="375" alt="image" src="https://user-images.githubusercontent.com/63265930/173683058-519ee14a-e970-440e-a367-e56a830a991d.png">

The database, however, contains data in all columns. To understand the relationship of the variables between them. A correlation analysis was performed, with the table presented in the Appendix (Graph B), wich gives us an overview of the interaction of the variables.

<img width="571" alt="image" src="https://user-images.githubusercontent.com/63265930/173683177-baf8fbf4-184c-4686-ab52-ae3de7b6ca6e.png">

We dropped the “ID” column from the original data since it is unrelated. The new dataset was then divided into a training set(202000 samples) and a holdout testing set(50000 samples).

## Modeling
### Summary
Our best model yet is the Random Forest model, with nodesize=5, ntree=500 and mtry=4. The AUC of the model is 0.9387 and the OOS Accuracy is 0.875104. We set the threshold to 0.1, which gives us the best result for our business problem - focusing on targeting high true positive rate. By setting the threshold to 0.1 in the model, we eliminated 99.6 percent of possible loan defaults while keeping the mistakes at an 21.3% acceptable level. The advantage for this model is that it gives very accurate predictions in comparison to other models we tested. One disadvantage is that this model is computationally inefficient. If the client want to modify our model or scale it onto a larger dataset, it would be computationally challenging.

<img width="649" alt="image" src="https://user-images.githubusercontent.com/63265930/173684011-4360e64e-cef7-4355-8697-7aecb77aa8e2.png">

Our best alternative is a Logistic Regression model. The AUC of the model is 0.6518, out-of-sample accuracy is around 0.7897643 (see the ROC curve in down below). In comparison to Random Forest, the Logistic Regression takes a shorter time to run, but the prediction accuracy is also much lower.

<img width="510" alt="image" src="https://user-images.githubusercontent.com/63265930/173683961-c6387b8c-b1a3-4619-a561-2f5bde5ac6bf.png">

### Modeling Process
We first started with K-means clustering. When we tried to select the k for K-means, we introduced the idea of regularization via Lasso Regression (L1 Regularization). Based on the HDIC result, our most optimal k value is 17. However, both BIC and AIC have optimal k values over 50, which may lead to severe overfitting since our dataset contains over 252,000 observations.

K-means clustering is an unsupervised method, and we would like to go on and see other supervised methods before jumping to an conclusion.

We then ran the Principal Component Analysis. PCA, however, does not work very effectively in our case. After reducing the dimentionality of the variables, we still ended up with a lot of factors that were able to explain the variance.

<img width="658" alt="image" src="https://user-images.githubusercontent.com/63265930/173684936-a1c3146a-bb20-4b89-8644-e819d2cd016f.png">

If we take a closer look at all principal components, we also realized that they are all complicated combinations of large numbers of features, which certainly does not help us to build the model in a more efficient manner.

We shifted our attention to Classification Tree and Regressions. We first ran a Cross Validation for out-of-sample R^2 and out-of-sample accuracy on Classification Tree, Lasso Regression, Post Lasso, and Logistic Regression. 

<img width="689" alt="image" src="https://user-images.githubusercontent.com/63265930/173929423-b81fcd8a-0c32-4677-8ab2-fae8657ca222.png">

<img width="473" alt="image" src="https://user-images.githubusercontent.com/63265930/173929465-bfde8ab6-acb2-4ebe-b953-c7729b16c21e.png">

Our Random Forest, though have a very high accuracy, was deemed too time-consuming and not realistic to be included in this step due to limitations on our computational power. All models performed well in the OOS Accuracy test. In the OOS R^2 test, however, the numbers are relatively low. We kept that in mind and continued to test the individual performance of models from the ROC perspective.

For the Classification Tree, the initial model with all variables failed to deliver a satisfactory result. The AUC of the model was 0.5, which indicates that it is no better than a null model. We tried to use different combinations of independent variables, but nothing seemed to work to improve the performance of the model (see the ROC curve below).

<img width="471" alt="image" src="https://user-images.githubusercontent.com/63265930/173930013-4ff209ae-b938-4b8f-8cbc-b67675a6eef0.png">

We also tested Lasso and Post-Lasso. The OOS accuracies were 0.7898 and 0.7901, respectively; the AUCs were 0.6363 and 0.6367, respectively. These results
are not very ideal.

By far, the best candidate is the Logistic Regression. The OOS accuracy is 0.7901 and the AUC is 0.6367. Since it is only slightly better than Post-Lasso, our team decided to make a final attempt using Random Forest.

For the Random Forest, we used nodesize=5, ntree=500, and mtry=4. The result was encouraging - the model efficiently identified those who are more likely to default on their loans and flagged them, giving the banks another line of defense when it comes to personal loan applications.

## Evaluation
The result of the data mining process was evaluated by TPR (True Positive Rate) and FPR (False Positive Rate). A high TPR and low FPR would suggest that the predictive model is well-performing.

Because of our high TPR rate (0.96), nearly all risky customers will be given the risk flag. Serving with risky customers is costly since defaults will cause huge loss to the
bank. Dealing with default is also time-consuming for the bank. Therefore reducing the
risk will definitely help the bank on not only cost but also time. With the precise
prediction on that, banks can reduce a lot of cost on the default customers since they all
have risk flags.





