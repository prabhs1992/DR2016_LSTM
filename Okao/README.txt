General Notes:
All scripts run and display all kinds of accuracies, ( framewise, teamwise and speaker wise accuracies).
For the first iteration of CV the script uses the values for parameters that gave me the best accuracy while testing on my computer. So I'm running it 51 times.(the remaining 50 iterations' values are randomly chosen)
The script is using a .txt file for the scripts to communicate with each other(so it works if both the scripts required for CV are placed in the same location).
The scipts take mean of every 5 frames.

Order of videos processed by cross validation script of feature: The videos are sorted according to a permutation.txt kept in the same directory as the code before train/validation/test split is performed. If this permutation.txt is not present, then the video list is first sorted according to the name and then the split is performed. This permutation.txt can thus be used to make sure that the same split is performed for train/validation/test across different feature scripts.

Order of videos processed by standalone script: The videos are sorted randomly before train/validation/test split is performed.

--------------------------------------------------------------------------------------------------------------------------------------------
Instructions to run:
1. lstm_okao_mean_cv.py is a python script to perform CV, that runs another script- validator.py multiple times for validation and testing. To run this please change the following variables as necessary:
lines 9-12 of lstm_okao_mean_with_cv:
validation_runs = 51
valid_cnt = 5
test_cnt = 5
limit = 36
validation_runs is number of times CV needs to be performed.
valid_cnt is the number of videos to be validated on. (count starts after the videos that are trained on i.e. limit-valid_cnt-test_cnt)
test_cnt is the number of videos to be tested on. (count starts after the videos that are trained and validated on i.e. limit-test_cnt)
limit is the total no. of videos to be trained+tested on from the okao directory. 
Also lines 17-23 of the above file can be altered to specify the range of each parameter value to be selected from.

and lines 310-311 of validator_okao.py:
wb = load_workbook('/home/prabhanjan/Downloads/Matlab_MetaData.xlsx', read_only=True)
dir = '/media/prabhanjan/New/DR/Okao Features/'
where wb is the location of the Matlab_Metadata.xlsx and dir is the directory path to okao features (all okao files should be in 1 single directory). 

Order of videos: The video list is sorted according to a permutation.txt (kept in the same directory as the code) before the train/validation/test split is performed. If this permutation.txt is not present, then the video list is first sorted by the name and then the split is performed. This permutation.txt can thus be used to make sure that the same split is performed for train/validation/test across different feature scripts.

--------------------------------------------------------------------------------------------------------------------------------------------
1. lstm_okao_mean.py is a standalone code that runs lstm on the okao features without doing CV. To run this please change the following variables as necessary:
lines 331-334 (starting lines inside the main() function):
wb = load_workbook('/home/prabhanjan/Downloads/Matlab_MetaData.xlsx', read_only=True)
dir = '/media/prabhanjan/New/DR/Okao Features/'
testcnt = 5
limit = 36
where wb is the location of the Matlab_Metadata.xlsx and dir is the directory path to okao features (all okao files should be in 1 single directory). testcnt is the count of the files to be tested. limit is the no. of videos to be trained+tested on from the okao directory. 

Order of videos: The videos are sorted randomly before train/validation/test split is performed.

--------------------------------------------------------------------------------------------------------------------------------------------
After the changes, the scripts should run (no arguments required, just python script_name.py). Arguments can be specified too(check the flags in the code)

