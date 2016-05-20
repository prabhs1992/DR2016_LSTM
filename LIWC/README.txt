There are 2 scripts that use decision trees and perform classification on the LIWC data.

Instructions to run:
-------------------------------------------------------------------------------------------------------------------------------------------
1. liwc_decision_tree_loocv.py
This script performs the leave one out cross validation on the LIWC features.
To run this script change the lines 12-14 in the script:
	wb = load_workbook('/home/prabhanjan/Downloads/Matlab_MetaData.xlsx', read_only=True)
	liwc_data = '/media/prabhanjan/New/DR/LIWC Features/LIWC_Data.xlsx'
	limit = 288
where wb is the directory where the MetaData is stored.
liwc_data is the location where the LIWC_Data.xlsx is stored.
limit is the no.of lines in the excel that need to be considered as part of training+testing set from the excel data. (This can be easily scripted to pick the total lines from the excel, but has not been done to provide flexibility of choosing the sample lines from the data)

-------------------------------------------------------------------------------------------------------------------------------------------
2. liwc_decision_tree.py
This script performs classification on the test data after fitting the decision tree model on train data.
To run the script edit lines 12-16:
	wb = load_workbook('/home/prabhanjan/Downloads/Matlab_MetaData.xlsx', read_only=True)
	liwc_data = '/media/prabhanjan/New/DR/LIWC Features/LIWC_Data.xlsx'
	testcnt = 80
	limit = 288
where wb is the directory where the MetaData is stored.
liwc_data is the location where the LIWC_Data.xlsx is stored.
limit is the no.of lines in the excel that need to be considered as part of training+testing set from the excel data. (This can be easily scripted to pick the total lines from the excel, but has not been done to provide flexibility of choosing the sample lines from the data)
testcnt is the no of data samples to test on from the entire data.

-------------------------------------------------------------------------------------------------------------------------------------------Note: In both scripts before fitting the model, the data set is shuffled randomly. (In script 2 after shuffling randomly, the last testcnt number of data is used for testing)
