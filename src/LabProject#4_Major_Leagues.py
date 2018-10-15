
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df_nba_elo=pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\Hw4\\3_nba_elo_Lab_Project#4\\nba_elo.csv')


# In[3]:


df_nba_elo.info()


# In[4]:


df_nba_elo.describe()


# In[6]:


df_nba_elo.dtypes


# In[23]:


#addtiomnal info1: Total number of matches played each day/ or total days on which match was held
df_nba_elo.groupby(['date' ]).count()['season']


# In[24]:


df_nba_elo_No_of_matches_each_day=df_nba_elo.groupby(['date' ]).count()['season']
df_nba_elo_No_of_matches_each_day= df_nba_elo_No_of_matches_each_day.to_frame()
df_nba_elo_No_of_matches_each_day.head()


# In[27]:


df_nba_elo_No_of_matches_each_day=df_nba_elo_No_of_matches_each_day.sort_values('season',ascending=False)


# In[29]:


# Maximum matches that got played on 5 day and where 15
df_nba_elo_No_of_matches_each_day.head(10)


# In[38]:


df_nba_elo_No_of_matches_each_day = df_nba_elo_No_of_matches_each_day.rename(columns={'Number of Matche Played': 'Number of Match Played'})


# In[39]:


df_nba_elo_No_of_matches_each_day.head(10)


# In[40]:


df_nba_elo_No_of_matches_each_day.to_csv('AdditionalInfo#1_Count_Of_Match_Played_dateWise.csv', sep=',')


# In[42]:


#addtiomnal info2: Total number of matches played in each season 
df_nba_elo.groupby(['season' ]).count()['date']


# In[44]:


df_nba_elo_Match_per_season=df_nba_elo.groupby(['season' ]).count()['date']
df_nba_elo_Match_per_season= df_nba_elo_Match_per_season.to_frame()
df_nba_elo_Match_per_season.head()
df_nba_elo_Match_per_season=df_nba_elo_Match_per_season.sort_values('date',ascending=False)


# In[47]:


# maximum number of matches were 1319 in season 2014 and 2006
df_nba_elo_Match_per_season.head(10)
df_nba_elo_Match_per_season = df_nba_elo_Match_per_season.rename(columns={'date': 'Number of Matches'})
df_nba_elo_Match_per_season.head(10)


# In[48]:


df_nba_elo_No_of_matches_each_day.to_csv('AdditionalInfo#2_SeasonWise_MatchCount.csv', sep=',')


# In[51]:


#addtiomnal info3: To find the mean score for all the team1
df_nba_elo.groupby(['team1' ]).mean()['score1']


# In[52]:


df_nba_elo_MeanScore_Team1=df_nba_elo.groupby(['team1' ]).mean()['score1']
df_nba_elo_MeanScore_Team1= df_nba_elo_MeanScore_Team1.to_frame()
df_nba_elo_MeanScore_Team1.head()


# In[53]:


df_nba_elo_MeanScore_Team1=df_nba_elo_MeanScore_Team1.sort_values('score1',ascending=False)


# In[54]:


df_nba_elo_MeanScore_Team1.head(10)


# In[55]:


# maximum mean score is 125.132653  for team DNA
df_nba_elo_MeanScore_Team1.head(10)
df_nba_elo_MeanScore_Team1 = df_nba_elo_MeanScore_Team1.rename(columns={'score1': 'Mean Score'})
df_nba_elo_MeanScore_Team1.head(10)


# In[56]:


df_nba_elo_MeanScore_Team1.to_csv('AdditionalInfo#3_MeanScore_for_AllTeam1.csv', sep=',')


# In[57]:


#addtiomnal info4: To find the mean score for all the team2
df_nba_elo.groupby(['team2' ]).mean()['score2']


# In[60]:


## maximum mean score is 125.132653  for team DNA
df_nba_elo_MeanScore_Team2=df_nba_elo.groupby(['team2' ]).mean()['score2']
df_nba_elo_MeanScore_Team2= df_nba_elo_MeanScore_Team2.to_frame()
df_nba_elo_MeanScore_Team2.head()
df_nba_elo_MeanScore_Team2 = df_nba_elo_MeanScore_Team2.rename(columns={'score2': 'Mean Score'})
df_nba_elo_MeanScore_Team2.head(10)


# In[61]:


df_nba_elo_MeanScore_Team2=df_nba_elo_MeanScore_Team2.sort_values('Mean Score',ascending=False)


# In[62]:


## Team2: maximum mean score is 120.421053  for team WSA
df_nba_elo_MeanScore_Team2.head(10)


# In[63]:


df_nba_elo_MeanScore_Team2.to_csv('AdditionalInfo#4_MeanScore_for_AllTeam2.csv', sep=',')


# In[65]:


#Additiomnal info5: to access/get date, month and year information from date column


# In[66]:


df_nba_elo_timeStamp.head(10)


# In[68]:


df_nba_elo_new= df_nba_elo


# In[70]:


df_nba_elo_new.head()


# In[73]:


df_nba_elo_new.info()


# In[74]:


#created 3 new column for year month and date
df_nba_elo_new['year'] = pd.DatetimeIndex(df_nba_elo_new['date']).year
df_nba_elo_new['month'] = pd.DatetimeIndex(df_nba_elo_new['date']).month
df_nba_elo_new['date'] = pd.DatetimeIndex(df_nba_elo_new['date']).day


# In[75]:


df_nba_elo_new.head()


# In[76]:


df_nba_elo_new.drop(['neutral', 'playoff','carmelo1_pre','carmelo2_pre', 'carmelo1_post', 'carmelo2_post','carmelo_prob1','carmelo_prob2'], axis=1, inplace=True)


# In[77]:


df_nba_elo_new.head()


# In[78]:


#Additiomnal info#6: The year in which maximum number of games were played.
df_nba_elo_newYEAR=df_nba_elo_new.groupby(['year']).count()['team1']


# In[85]:


df_nba_elo_newYEAR.head()
df_nba_elo_newYEAR= df_nba_elo_newYEAR.to_frame()
df_nba_elo_newYEAR = df_nba_elo_newYEAR.rename(columns={'team1': 'Number of Matches Played'})


# In[86]:


df_nba_elo_newYEAR = df_nba_elo_newYEAR.rename(columns={'team1': 'Number of Matches Played'})


# In[87]:


df_nba_elo_newYEAR.head()


# In[88]:


df_nba_elo_newYEAR=df_nba_elo_newYEAR.sort_values('Number of Matches Played',ascending=False)


# In[89]:


#in year 2012 maximum number of 1474 matches were played
df_nba_elo_newYEAR.head()


# In[90]:


df_nba_elo_newYEAR.to_csv('AdditionalInfo#6_NumberOfMatchesPlayedYearWise.csv', sep=',')


# In[94]:


#Additiomnal info#6: The month in which maximum number of games were played.
df_nba_elo_newMonth=df_nba_elo_new.groupby(['month']).count()['team1']
df_nba_elo_newMonth.head()
df_nba_elo_newMonth= df_nba_elo_newMonth.to_frame()
df_nba_elo_newMonth = df_nba_elo_newMonth.rename(columns={'team1': 'Number of Matches Played'})
df_nba_elo_newMonth = df_nba_elo_newMonth.rename(columns={'team1': 'Number of Matches Played'})
df_nba_elo_newMonth=df_nba_elo_newMonth.sort_values('Number of Matches Played',ascending=False)


# In[96]:


# maximum number of matches were played in month of March followed by January,December, November and Feb.
# Least favourite month : June
df_nba_elo_newMonth.head(12)


# In[97]:


df_nba_elo_newMonth.to_csv('AdditionalInfo#7_NumberOfMatchesPlayedMonthWise.csv', sep=',')


# In[101]:


#Additiomnal info#7: The date in which maximum number of games were played.
df_nba_elo_newDate=df_nba_elo_new.groupby(['date']).count()['team1']
df_nba_elo_newDate.head()
df_nba_elo_newDate= df_nba_elo_newDate.to_frame()
df_nba_elo_newDate = df_nba_elo_newDate.rename(columns={'team1': 'Number of Matches Played'})
df_nba_elo_newDate = df_nba_elo_newDate.rename(columns={'team1': 'Number of Matches Played'})
df_nba_elo_newDate=df_nba_elo_newDate.sort_values('Number of Matches Played',ascending=False)


# In[110]:


# Maximum match were played/held during start or end of the month
df_nba_elo_newDate.head(30)


# In[111]:


df_nba_elo_newDate.to_csv('AdditionalInfo#8_NumberOfMatchesPlayedDateWise.csv', sep=',')


# In[113]:


df_nba_elo.head()


# In[114]:


df_nba_elo_modified=df_nba_elo


# In[117]:


df_nba_elo.info()


# In[118]:


df_nba_elo_modified["team1"] = df_nba_elo_modified["team1"].astype('category')
df_nba_elo_modified.dtypes
df_nba_elo_modified["team1_cat"] = df_nba_elo_modified["team1"].cat.codes
df_nba_elo_modified.head()


# In[119]:


df_nba_elo_modified["team2"] = df_nba_elo_modified["team2"].astype('category')
df_nba_elo_modified.dtypes
df_nba_elo_modified["team2_cat"] = df_nba_elo_modified["team2"].cat.codes
df_nba_elo_modified.head()


# In[120]:


df_nba_elo_modified.drop(['team1', 'team2'], axis=1, inplace=True)


# In[124]:


df_nba_elo_modified.head()


# In[125]:


df_nba_elo_modified.describe()


# In[127]:


df_nba_elo_modified.shape


# In[128]:


df_nba_elo_modified_2=df_nba_elo_modified


# In[129]:


df_nba_elo_modified_2.head()


# In[130]:


# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(df_nba_elo_modified['score1'])
# Remove the labels from the features
# axis 1 refers to the columns
df_nba_elo_modified= df_nba_elo_modified.drop('score1', axis = 1)
# Saving feature names for later use
feature_list = list(df_nba_elo_modified.columns)
# Convert to numpy array
df_nba_elo_modified = np.array(df_nba_elo_modified)


# In[131]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_df_nba_elo_modified, test_df_nba_elo_modified, train_labels, test_labels = train_test_split(df_nba_elo_modified, labels, test_size = 0.25, random_state = 42)


# In[132]:


print('Training Features Shape:', train_df_nba_elo_modified.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_df_nba_elo_modified.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[133]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_df_nba_elo_modified, train_labels);


# In[135]:


# Use the forest's predict method on the test data
predictions = rf.predict(test_df_nba_elo_modified)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[136]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

