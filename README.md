# HW-4_Major-Leagues
EECS 731:Assignment 4: HW#4_Major Leagues

Notebook: LabProject#4_Major_Leagues.ipynb Purpose: Deduced Additional Information and random forest regression analysis to find the score for each team.

* Loaded the nba_elo.csv in a dataframes.

* Additional Information #1: To find total number of matches played each day/ or total days on which match was held/played.

Converted the above new data into a new data frame (df_nba_elo_No_of_matches_each_day).

Saved the resulted dataframe in a csv (AdditionalInfo#1_Count_Of_Match_Played_dateWise.csv)

* Additional Information #2: To find total number of matches played in each season.

Converted the above new data into a dataframe (df_nba_elo_Match_per_season).

Saved the resulted dataframe in a csv (AdditionalInfo#2_SeasonWise_MatchCount.csv).

* Additional Information #3: To find the mean score for all the team1.

Converted the above new data into a dataframe (df_nba_elo_MeanScore_Team1)

Saved the resulted dataframe in a csv (AdditionalInfo#3_MeanScore_for_AllTeam1.csv)

* Additional Information #4: To find the mean score for all the team2.

Converted the above new data into a dataframe (df_nba_elo_MeanScore_Team2)

Saved the resulted dataframe in a csv (AdditionalInfo#4_MeanScore_for_AllTeam2.csv)

* Additional Information #5: Created 3 new columns namely year, month and date for each of the match played.

* Additional Information #6: To find for each year how many matches were played. 

Converted the above new data into a dataframe (df_nba_elo_newYEAR )

Saved the resulted dataframe in a csv (AdditionalInfo#6_NumberOfMatchesPlayedYearWise.csv)

* Additional Information #7: To find for each month how many matches were played. 

Converted the above new data into a dataframe (df_nba_elo_newMonth )

Saved the resulted dataframe in a csv (AdditionalInfo#7_NumberOfMatchesPlayedMonthWise.csv)

* Additional Information #8: To find for each month how many matches were played. 

Converted the above new data into a dataframe (df_nba_elo_newDate )

Saved the resulted dataframe in a csv (AdditionalInfo#8_NumberOfMatchesPlayedDateWise.csv)

My findings -->

* On dates 2013-04-17, 2016-11-25, 2014-04-16, 2009-01-02, 2011-04-13	maximum number of 15 matches were played.
* For season 2014 and 2016 maximum number of matches were playes i.e. 1319 matches.
* Among all the team1, DNA has the maximum mean score of 125.132653
* Among all the team2, WSA has the maximum mean score of 120.421053
* In year 2012, maximum number of 1474 matches were played.
* For all the seasons, the maximum number of matches were played in March i.e. 11877 matches.
* Maximum number of matches were played/held during start or end of the month.

Random Forest: Regression Analysis

* In order to apply random forest, changed the datatype of team1 and team2 column (object type) to int
* Find the labels and stored them separately i.e. the score we wanted to predict.
* Remove the labels from the features
* Saving feature names for later use.
* To convert the dataframe to numpy array.
* Used Skicit-learn to split data into training and testing sets.
* Imported the random forest model.
* Instantiated the model with 1000 decision trees
* Trained the model on training data
* Used the forest's predict method on the test data
* Calculated the absolute errors
* Printed the mean absolute error (mae) i.e. 5.16 degree
* Calculated mean absolute percentage error (MAPE)

*Accuracy: 94.71 %.*
