# Credit Card Fraud Detection

### **About the bank/client:**

`Finex` is a  leading financial service provider based out of Florida, US. It offers a wider range of producyts and business services to the customers through different channels like in-person banking, ATMs and online banking.

### **Issue the client is facing**
Over the last few years, `Finex` has observed that a significantly large number of unauthorised transactions are being made, due to which the bank has been facing a huge revenue and profitability crisis. Many customers have been complaining about unauthorised transactions being made through their credit/debit cards. It has been reported that fraudsters use stolen/lost cards and hack private systems to access the personal and sensitive data of many cardholders. They also indulge in ATM skimming at various POS terminals such as gas stations, shopping malls, and ATMs that do not send alerts or do not have OTP systems through banks. 

Such fraudulent activities have been reported to happen during non-peak and odd hours of the day leaving no room for suspicion.

In most cases, customers get to know of such unauthorised transactions happening through their cards quite late as they are unaware of such ongoing credit card frauds or they do not monitor their bank account activities closely. This has led to late complaint registration with `Finex` and by the time the case is flagged fraudulent, the bank incurs heavy losses and ends up paying the lost amount to the cardholders.

Now, `Finex` is also not really equipped with the latest financial technologies, and it is becoming difficult for the bank to track these data breaches on time to prevent further losses.

For `Finex`, retaining high profitable customers is the most important business goal. With the rise in digital payment channels, banking frauds, however poses a significant threat to this goal for many banks.


### **Problem to be Solved**

> `1) Finding when a fraud has occured.`

> `2) Making a `

<br><br>

----------------------
## <font color='red'> **Part 1: Root Cause Analysis**</font>
----------------------

As a consultant hired by `Finex`, our task is to identify the root cause of this issue of unauthorized transactions on credt card/debit card and recomment ways to mitigate this problem. They also want us to find the action areas to come up with a long-term solution that would help the bank generate high revenue with minimal losses.

Understanding the pipeline for a typical transaction and the challenges at each of these steps of the transaction process so that you can make appropriate interventions to solve the problem. 

#### ***What we need to do?***
- Identify and stop the fraudulent transactions that they (fraudsters) are making.

#### ***What we DON't need to do?***
- Apprehending fraudsters is outside the scope of banking operations.
- Not stop the fraudsters

We would need to conduct the **`root cause analysis`** with the bank’s Person of Contact to understand the processes/structures that are already in place to deal with anomalous transactions. 

<br><br>

### **The 5W + How Framework**
**`Q1.	Who is involved in the process?`** <br>
***Perpetrator***
- The thief/The credit card fraudster. They can be individuals or an organization.

***Victims***
- The customer
- The Bank

**`Q2.	What do they do with it?`** <br>
They take the money for their personal gains and their leisure lifestyle. To them this is a business.

**`Q3.	Where do the transactions happen?`**<br>
These transactions tend to happen generally at:
- Grocery Stores
- Gas Stations
- Restaurants
- Shopping outlets
- ATMs (for debit card frauds)

**`Q4.	When does it happen?`**<br>
It can happen at anytime. But, there might be certain times where the frequency of Credit card frauds are high like - "Holiday Season - Black Friday Sales", "During night hours".

**`Q5.	How is business affected by this?`**<br>
As a result the banks tend to loose millions of dollars every year. They suffer huge revenue and profitability crisis.




<br><br>

----------------------
## <font color='red'> **Part 2: Data Analysis and Modelling** </font>
----------------------
Given all possible hypotheses and considering the feasibility and customer time, the most suitable solution is to implement a `fraud detection system`. This does not affect the customer’s time with extra OTP checks on all transactions and is also quite feasible, as educating all customers on various fraudulent techniques is a challenging task. 

**Building a fraud detection system is a one time procedure and deploying this would be a permanent resolution to the long time blocker that the banks have been facing since years.**

In the banking industry, detecting credit card fraud using machine learning is not just a trend; it is a necessity for the banks, as they need to put proactive monitoring and fraud prevention mechanisms in place. Machine learning helps these institutions reduce time-consuming manual reviews, costly chargebacks and fees, and denial of legitimate transactions.


- Suppose you are part of the analytics team working on a fraud detection model and its cost-benefit analysis. 

- You need to develop a machine learning model to detect fraudulent transactions based on the historical transactional data of customers with a pool of merchants. 

- You can learn more about transactional data and the creation of historical variables from the link attached here. 

- You may find this helpful in the capstone project while building the fraud detection model. 

- Based on your understanding of the model, you have to analyse the business impact of these fraudulent transactions and recommend the optimal ways that the bank can adopt to mitigate the fraud risks.


### **Understanding and Defining Fraud**

Credit card fraud is any dishonest act or behaviour to obtain information without the proper authorisation of the account holder for financial gain. Among the different ways of committing fraud, `skimming` is the most common one. `Skimming` is a method used for duplicating information located on the magnetic stripe of the card.  Apart from this, other ways of making fraudulent transactions are as follows:

- Manipulation or alteration of genuine cards
- Creation of counterfeit cards
- Stolen or lost credit cards
- Fraudulent telemarketing

### **Data Understanding**

Data is Present at: https://www.kaggle.com/datasets/kartik2112/fraud-detection

The data set contains credit card transactions of around 1,000 cardholders with a pool of 800 merchants from 1 Jan 2019 to 31 Dec 2020. It contains a total of 18,52,394 transactions, out of which 9,651 are fraudulent transactions. The data set is highly imbalanced, with the positive class (frauds) accounting for 0.52% of the total transactions. Now, since the data set is highly imbalanced, it needs to be handled before model building. The feature 'amt' represents the transaction amount. 

**Target Variable - "is_fraud"** <br>
The feature 'is_fraud' represents class labelling and takes the following values:
- 1 -->  The transaction is a fraudulent transaction and 
- 0 -->  Otherwise.

<br> 

### **Project Pipeline**
The project pipeline can be briefly summarised in the following steps:

#### **Step 1 - Understanding Data:** 
- In this step, you need to load the data and understand the features present in it. 
- This will help you choose the features that you need for your final model.

#### **Exploratory data analytics (EDA):** 
- Normally, in this step, you need to perform univariate and bivariate analyses of the data, followed by feature transformations, if necessary. 
- You can also check whether or not there is any skewness in the data and try to mitigate it, as skewed data can cause problems during the model-building phase.

#### **Train/Test Data Splitting:**
- In this step, you need to split the data set into training data and testing data in order to check the performance of your models with unseen data. 
- You can use the stratified k-fold cross-validation method at this stage. 
- For this, you need to choose an appropriate k value such that the minority class is correctly represented in the test folds.

#### **Model Building or Hyperparameter Tuning:**
- This is the final step, at which you can try different models and fine-tune their hyperparameters until you get the desired level of performance out of the model on the given data set. 
- Ensure that you start with a baseline linear model before going towards ensembles. 
- You should check if you can get a better performance out of the model by using various sampling techniques.

#### **Model Evaluation:** 
- Evaluate the performance of the models using appropriate evaluation metrics. 
- Note that since the data is imbalanced, it is important to identify which transactions are fraudulent transactions more accurately than identifying non-fraudulent transactions. 
- Choose an appropriate evaluation metric that reflects this business goal.
 
#### **Business Impact:** 
After the model has been built and evaluated with the appropriate metrics, you need to demonstrate its potential benefits by performing a cost-benefit analysis which can then be presented to the relevant business stakeholders. 


