# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:03:33 2016

@author: Marcia
"""

import pandas as pd
#Read in BasePatsMerged.
all_final_df = pd.read_csv('/Users/Marcia/OneDrive/DSBA 6156 ML/CMS Project/DataFiles For Python/Final_161120.csv',
                         keep_default_na=False, na_values=[""])

#Decided to use "copy" to make a complete new copy of the original df, not just a reference to it.
copy_all_final_df=all_final_df.copy()

copy_all_final_df.columns
#Want to keep just two fields, the patnum and the maingroup
#data=copy_all_final_df[['NC_Licenced_teacher', 't_03_exp',
#       't_410_exp', 't_11p_exp', 'adv_deg', 'one_yr_teacher_turnover',
#       'lateral_entry_t','Grade_binary']]
data=copy_all_final_df[['crime_per_100_stud', 'short_susp_per_c_num',
       'long_susp_per_c_num', 'expelled_per_c_num', 'Grade_binary']]       
#Shorten column names.
#data.rename(columns = {'NC_Licenced_teacher':'Lic', 't_03_exp':'t03',
#       't_410_exp':'t410', 't_11p_exp':'t11', 'adv_deg':'ad', 'one_yr_teacher_turnover':'one',
#       'lateral_entry_t':'lat','Grade_binary':'class'}, inplace=True)       
data.rename(columns = {'crime_per_100_stud':'crime', 'short_susp_per_c_num':'short',
        'long_susp_per_c_num':'long', 'expelled_per_c_num':'expell', 
       'Grade_binary':'class'}, inplace=True)  

#Print descriptive statistics
print(data.describe())

import matplotlib.pyplot as plt
#pd.options.display.mpl_style = 'default'
#data.groupby('class').hist()
#data.groupby('class').ad.hist(alpha=0.5)
#plt.title('Advanced Degree: A/B=Blue, Else=Pink')
#plt.figure()
#data.groupby('class').one.hist(alpha=0.5)
#plt.title('One Year Teacher Turnover: A/B=Blue, Else=Pink')
#plt.figure()
#data.groupby('class').Lic.hist(alpha=0.5)
#plt.title('NC Licenced Teacher: A/B=Blue, Else=Pink')
#plt.figure()
#data.groupby('class').t03.hist(alpha=0.5)
#plt.title('Teacher with less than 4 years of Experience: A/B=Blue, Else=Pink')
#plt.figure()
#data.groupby('class').t410.hist(alpha=0.5)
#plt.title('Teacher with 4-10 years of Experience: A/B=Blue, Else=Pink')
#plt.figure()
#data.groupby('class').t11.hist(alpha=0.5)
#plt.title('Teacher with more than 10 years of Experience: A/B=Blue, Else=Pink')
#plt.figure()
#data.groupby('class').lat.hist(alpha=0.5)
#plt.title('Teacher With Lateral Entry: A/B=Blue, Else=Pink')
#plt.figure()
from pandas.tools.plotting import scatter_matrix
scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.figure()
data.groupby('class').crime.hist(alpha=0.5)
plt.title('Crime Per 100 Students: A/B=Blue, Else=Pink')
plt.figure()
data.groupby('class').short.hist(alpha=0.5)
plt.title('Short Term Suspensions: A/B=Blue, Else=Pink')
plt.figure()
data.groupby('class').long.hist(alpha=0.5)
plt.title('Long Term Suspensions: A/B=Blue, Else=Pink')
plt.figure()
data.groupby('class').expell.hist(alpha=0.5)
plt.title('Expellsions: A/B=Blue, Else=Pink')
plt.figure()