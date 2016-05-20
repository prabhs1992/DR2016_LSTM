# prabhanjan_bhat_DR2016

Hi Professor,

PFA the final code for all the features.
The changes from the last submitted code:
1. Added random shuffling to features' video list via an external file as suggested:
The video list is sorted according to a permutation.txt (kept in the same directory as the code) before the train/validation/test split is performed. If this permutation.txt is not present, then the video list is first sorted by the name and then the split is performed. This permutation.txt can thus be used to make sure that the same split is performed for train/validation/test across different feature scripts.

2. Added README.txt explaining instructions to run script for all features.
3. Added comments inline.

I'm not currently printing the results to a file, since I'm not sure whether the number of predicted labels would be same across different features. Should I print the results?

Thanks,
Prabhanjan

