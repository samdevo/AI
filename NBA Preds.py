#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import numpy as np


# In[178]:


players = pd.read_csv("./nba-players-stats/Players.csv")
season_stats = pd.read_csv("./nba-players-stats/Seasons_Stats.csv")


# In[179]:




# In[180]:


player_avg = season_stats.groupby("Player").mean()


# In[181]:


player_avg = player_avg.reset_index()
player_avg = player_avg.fillna(player_avg.mean())


# In[182]:


player_avg.sort_values(by='PTS', ascending=False)
from sklearn.neighbors import NearestNeighbors


# In[183]:


player_avg.columns


# In[184]:


columns = ['Player','G', 'GS', 'MP', 'PER',
       'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'OWS', 'DWS', 'WS', 'WS/48', 'blank2', 'OBPM',
       'DBPM', 'BPM', 'VORP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P',
       '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST',
       'STL', 'BLK', 'TOV', 'PF', 'PTS']


# In[185]:


player_avg = player_avg[columns].set_index("Player")


# In[186]:


player_avg = player_avg.fillna(0)


# In[188]:


names = player_avg.index.tolist()


# In[ ]:





# In[190]:


x = player_avg.reset_index().loc[player_avg.reset_index()['Player']=="Michael Jordan*"].set_index("Player")


# In[192]:


from sklearn.neighbors import DistanceMetric
dist = DistanceMetric.get_metric('euclidean')


# In[193]:


dist.pairwise(player_avg.reset_index().loc[player_avg.reset_index()['Player']=="Michael Jordan*"].set_index("Player"), player_avg.reset_index().loc[player_avg.reset_index()['Player']=="Carmelo Anthony"].set_index("Player"))




# In[194]:


x = player_avg.reset_index().loc[player_avg.reset_index()['Player']=="Michael Jordan*"].set_index("Player")


# In[195]:


x.to_numpy()


# In[196]:


from sklearn import preprocessing


# In[212]:


normalized_avg=(player_avg-player_avg.mean())/player_avg.std()
normalized_avg = normalized_avg.drop(["blank2"], axis=1)


# In[247]:





# In[299]:


closest = []

maxlen = 10
playerName = "Kyrie Irving"
dataset = normalized_avg #either this or player_avg
x = dataset.reset_index().loc[dataset.reset_index()['Player'] == playerName].set_index("Player")
for row in dataset.iterrows():
# #     print(row[0])
#     print(np.asarray(row))
    rowdist = dist.pairwise(x.to_numpy(), np.asarray(row[1:]))[0][0]
#     print(rowdist)
    if len(closest)<maxlen+1:
        closest.append((row[0], rowdist))
        closest.sort(key = lambda x: x[1])  
    elif rowdist < closest[maxlen-1][1]:
        closest.pop()
        closest.append((row[0], rowdist))
        closest.sort(key = lambda x: x[1])  
for player in closest[1:]:
    print(player[0])


# In[286]:


closest[1:]


# In[120]:


player_avg


# In[135]:


player_avg.index


# In[ ]:




