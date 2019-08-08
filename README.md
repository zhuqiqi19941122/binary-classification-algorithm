# Binary-classification-algorithm
Bank Precision Marketing Solutions-- using Logistic Regression and Tree Algorithms
Precision marketing makes sense for sellers and banks. Selecting a suitable user group for promotion, on the one hand, reduces the cost of promotion, and on the other hand increases the possibility of promotion success. In this project, we use a dataset from the [Bank Precision Marketing Solutions competiton](https://www.kesci.com/home/competition/5c234c6626ba91002bfdfdd3/content/0) on kesci and predict the probability of user purchase.
# Dataset
字段说明
NO	字段名称	 数据类型	  字段描述
1	    ID	      Int	     客户唯一标识
2	    age	      Int	     客户年龄
3	    job	      String	 客户的职业
4	   marital	  String   婚姻状况
5	  education	  String	 受教育水平
6	   default	  String	 是否有违约记录
7	   balance	  Int	     每年账户的平均余额
8	   housing	  String	 是否有住房贷款
9	    loan	    String	 是否有个人贷款
10	 contact	  String	 与客户联系的沟通方式
11	   day	    Int	     最后一次联系的时间（几号）
12	  month	    String	 最后一次联系的时间（月份）
13	 duration	  Int	     最后一次联系的交流时长
14	campaign	  Int	     在本次活动中，与该客户交流过的次数
15	  pdays	    Int	     距离上次活动最后一次联系该客户，过去了多久（999表示没有联系过）
16	previous	  Int	     在本次活动之前，与该客户交流过的次数
17	poutcome	  String	 上一次活动的结果
18	   y	      Int	     预测客户是否会订购定期存款业务
# Learning algorithms used
Logistic Regression
Support Vector Machine
Decision Trees
Random Forest
# Evaluation Methods
To evaluate the performance of each model, we used the ROC AUC Score. Learn more about ROC curves and AUC [here](https://www.dataschool.io/roc-curves-and-auc-explained/).
# Results
