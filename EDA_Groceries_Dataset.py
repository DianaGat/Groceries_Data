# -*- coding: utf-8 -*-
"""
"""
### Exploritary data analysis
### Diana Hilleshein
### Jul 28, 2022

### Data: <https://www.kaggle.com/code/pakomm/groceries-dataset-eda/data>

###data processing
#libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#load data
df = pd.read_csv("C:/datas/Groceries_dataset.csv")
df.info() #38765 obs, 3 cols
df.head()
#NAs
df.isnull().sum()
#transformations
df.Member_number = df.Member_number.astype("string")
df.itemDescription = df.itemDescription.astype("string")
df.Date = pd.to_datetime(df.Date, dayfirst = True)
#duplicates
df[df.duplicated(keep=False)].sort_values(by='Member_number')
#descriptive
df.describe()

######## Time series plots
#data transformation
from datetime import datetime as dt
df["Weekday"], df["Day"], df["Month"], df["Year"] = df.Date.dt.strftime("%A"), df.Date.dt.strftime("%d"), df.Date.dt.strftime("%m"), df.Date.dt.strftime("%Y")
df = df.sort_values(by = "Date")
df.head()
#group by date
#df_grouped_by_Date = df.groupby("Date").agg({"itemDescription": "count"}).reset_index()
df.set_index('Date',inplace=True)
df_day_resampled_item_count = df.resample('D')[['itemDescription']].count()

###Plot items sold per day, 2014 + 2015
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator
#style
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [30, 10]})
ax = df_day_resampled_item_count.plot(kind='line',figsize=(15,5),legend=None)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.set_ylim([0, df_day_resampled_item_count["itemDescription"].max()+4])
#mean and sds
#2014
from datetime import timedelta
x1, y1 = [df_day_resampled_item_count.index.min(), 
          df_day_resampled_item_count.index.mean()-timedelta(days=1)], [df_day_resampled_item_count[df_day_resampled_item_count.index < df_day_resampled_item_count.index.mean()].itemDescription.mean(), df_day_resampled_item_count[df_day_resampled_item_count.index < df_day_resampled_item_count.index.mean()].itemDescription.mean()]
mean1 = plt.plot(x1, y1, color = "k", alpha = 0.55)
#
x1, y1 = [df_day_resampled_item_count.index.min(), 
          df_day_resampled_item_count.index.mean()-timedelta(days=1)], [df_day_resampled_item_count[df_day_resampled_item_count.index < df_day_resampled_item_count.index.mean()].itemDescription.mean() + df_day_resampled_item_count[df_day_resampled_item_count.index < df_day_resampled_item_count.index.mean()].itemDescription.std(), df_day_resampled_item_count[df_day_resampled_item_count.index < df_day_resampled_item_count.index.mean()].itemDescription.mean() + df_day_resampled_item_count[df_day_resampled_item_count.index < df_day_resampled_item_count.index.mean()].itemDescription.std()]
sd1pos = plt.plot(x1, y1, color = "k", alpha = 0.45, linestyle = "--")
#
x1, y1 = [df_day_resampled_item_count.index.min(), 
          df_day_resampled_item_count.index.mean()-timedelta(days=1)], [df_day_resampled_item_count[df_day_resampled_item_count.index < df_day_resampled_item_count.index.mean()].itemDescription.mean() - df_day_resampled_item_count[df_day_resampled_item_count.index < df_day_resampled_item_count.index.mean()].itemDescription.std(), df_day_resampled_item_count[df_day_resampled_item_count.index < df_day_resampled_item_count.index.mean()].itemDescription.mean() - df_day_resampled_item_count[df_day_resampled_item_count.index < df_day_resampled_item_count.index.mean()].itemDescription.std()]
sd1neg = plt.plot(x1, y1, color = "k", alpha = 0.45, linestyle = "--")
#2015
x2, y2 = [df_day_resampled_item_count.index.mean()+timedelta(days=1), df_day_resampled_item_count.index.max()], [df_day_resampled_item_count[df_day_resampled_item_count.index > df_day_resampled_item_count.index.mean()].itemDescription.mean(), df_day_resampled_item_count[df_day_resampled_item_count.index > df_day_resampled_item_count.index.mean()].itemDescription.mean()]
mean2 = plt.plot(x2, y2, color = "k", alpha = 0.55)
#
x1, y1 = [df_day_resampled_item_count.index.mean()+timedelta(days=1), df_day_resampled_item_count.index.max()], [df_day_resampled_item_count[df_day_resampled_item_count.index > df_day_resampled_item_count.index.mean()].itemDescription.mean() + df_day_resampled_item_count[df_day_resampled_item_count.index > df_day_resampled_item_count.index.mean()].itemDescription.std(), df_day_resampled_item_count[df_day_resampled_item_count.index > df_day_resampled_item_count.index.mean()].itemDescription.mean() + df_day_resampled_item_count[df_day_resampled_item_count.index > df_day_resampled_item_count.index.mean()].itemDescription.std()]
sd2pos = plt.plot(x1, y1, color = "k", alpha = 0.45, linestyle = "--")
#
x1, y1 = [df_day_resampled_item_count.index.mean()+timedelta(days=1), df_day_resampled_item_count.index.max()], [df_day_resampled_item_count[df_day_resampled_item_count.index > df_day_resampled_item_count.index.mean()].itemDescription.mean() - df_day_resampled_item_count[df_day_resampled_item_count.index > df_day_resampled_item_count.index.mean()].itemDescription.std(), df_day_resampled_item_count[df_day_resampled_item_count.index > df_day_resampled_item_count.index.mean()].itemDescription.mean() - df_day_resampled_item_count[df_day_resampled_item_count.index > df_day_resampled_item_count.index.mean()].itemDescription.std()]
sd2neg = plt.plot(x1, y1, color = "k", alpha = 0.45, linestyle = "--")
#legend
from matplotlib.lines import Line2D
line_p1 = Line2D([0,1],[0,1],linestyle='-', color = "k", alpha = 0.75)
dashed_line = Line2D([0,1],[0,1],linestyle='--', color = "k", alpha = 0.65)
plt.legend([line_p1, dashed_line],["Mean", "SD +/-"], 
           fontsize = "medium", framealpha = 0.6, loc= 3)
plt.ylabel("Daily sales")
plt.title("Items sold per day (2014 - 2015)")
plt.show()

###Plot items sold per month, 2014 + 2015
#data transformation
df_day_resampled_month_item_count = df.resample('M')[['itemDescription']].count()
#same style
ax = df_day_resampled_month_item_count.plot(kind='line', 
                                            figsize=(15,5),legend=None)
ax.yaxis.set_major_locator(MultipleLocator(200))
ax.set_ylim([0, df_day_resampled_month_item_count["itemDescription"].max()+100])
plt.ylabel("Monthly sales")
plt.title("Items sold per month (2014 - 2015)")
plt.show()

###Plot items sold perd day of month 2014 vs 2015
#data transformation
df_grouped_by_day_and_year = df.groupby(["Day", "Year"], as_index = False).agg(
    Purchase = ("itemDescription", "nunique"))
#plot
#style
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [17, 6]})
fig, ax = plt.subplots()
#2014
ax.plot(df_grouped_by_day_and_year[df_grouped_by_day_and_year["Year"] == "2014"].Day,
        df_grouped_by_day_and_year[df_grouped_by_day_and_year["Year"] == "2014"].Purchase,
        color = "#6B538C", linewidth = 4, alpha = 0.8)
#2015
ax.plot(df_grouped_by_day_and_year[df_grouped_by_day_and_year["Year"] == "2015"].Day,
        df_grouped_by_day_and_year[df_grouped_by_day_and_year["Year"] == "2015"].Purchase,
        color = "#8C373B", linewidth = 4, alpha = 0.8)
#legend
line2014 = Line2D([0,1],[0,1],linestyle='-', color = "#6B538C", alpha = 1, linewidth = 4)
line2015 = Line2D([0,1],[0,1],linestyle='-', color = "#8C373B", alpha = 1, linewidth = 4)
plt.legend([line2015, line2014],["2014", "2015"], 
           fontsize = "medium", framealpha = 0.6)
plt.ylabel("Amount of purchases, items")
plt.xlabel("Day of a month")
plt.title("Amount of purchses per day of month,\n2014 vs 2015", size = 20)
plt.xlim([0,30])
plt.show()

#Plot items sold per month 2014 vs 2015
#data transformation
df_grouped_by_month_and_year = df.groupby(["Month", "Year"]).agg({"itemDescription": "count"}).reset_index()
monthly2014 = df_grouped_by_month_and_year[df_grouped_by_month_and_year["Year"] == "2014"][["Month", "itemDescription"]]
monthly2015 = df_grouped_by_month_and_year[df_grouped_by_month_and_year["Year"] == "2015"][["Month", "itemDescription"]]
#plot
#style
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [17, 6]})
#big plot
fig, ax = plt.subplots()
ax.plot(monthly2014.Month, monthly2014.itemDescription, color = "#6B538C", linewidth = 4, alpha = 0.8)
ax.plot(monthly2015.Month, monthly2015.itemDescription, color = "#8C373B", linewidth = 4, alpha = 0.8)
ax.set_xlim([monthly2014.Month.min(), monthly2014.Month.max()])
ax.yaxis.set_major_locator(MultipleLocator(100))
plt.ylabel("Monthly sales, items")
plt.xlabel("Date")
plt.title("Items sold per month,\n2014 vs 2015", size = 20)
ax.set_xticklabels(["Jan", "Feb", "Mar", "Aprl", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
#legend
line2014 = Line2D([0,1],[0,1],linestyle='-', color = "#6B538C", alpha = 1, linewidth = 4)
line2015 = Line2D([0,1],[0,1],linestyle='-', color = "#8C373B", alpha = 1, linewidth = 4)
plt.legend([line2015, line2014],["2015", "2014"], fontsize = "medium", framealpha = 0.6, bbox_to_anchor=(0.08,1.160))
#small plot
ax2 = fig.add_axes([.70, .76, .2, .25]) #[left, bottom, width, height]
ax2.plot(monthly2014.Month, monthly2014.itemDescription, color = "#6B538C")
ax2.plot(monthly2015.Month, monthly2015.itemDescription, color = "#8C373B")           
ax2.set_ylim(0, monthly2015.itemDescription.max() + 100)
ax2.set_xlim([monthly2014.Month.min(), monthly2014.Month.max()])
ax2.yaxis.set_major_locator(MultipleLocator(200))
ax2.set_xticklabels(["Jan", "Feb", "Mar", "Aprl", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.show()

###Barplot for sum items sold in 2014 and 2015
#data transformation
df_year_resampled_item_count = df.resample('Y')[['itemDescription']].count()
#plot
#style
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [6, 6]})
ax = sns.barplot(x = df_year_resampled_item_count.index, 
                 y = df_year_resampled_item_count.itemDescription,
                 alpha = 0.85)
ax.set_xticklabels([2014, 2015])
plt.title("Items sold, 2014 vs 2015")
plt.ylabel("Yearly sales, items")
plt.xlabel("")
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.30, p.get_height()-1100), size=12, color = "white", weight = "bold")
plt.show()

########## Items
# unique items 2014 vs 2015
df.groupby("Year").agg({"itemDescription": "nunique"}) 
# unique names of items 2014 vs 2015
set_grouped_by_year = df.groupby("Year").agg({"itemDescription": set}) 

###Top 15 items by  purchase frequency, 2014 vs 2015
#data transformation
df2014 = df[df.index < '2014-12-31 00:00:00'] #exept 2014-12-31, since the store did not work
df2015 = df[df.index > '2014-12-31 00:00:00']
value_counts_2014 = df2014["itemDescription"].value_counts()
value_counts_2015 = df2015["itemDescription"].value_counts()
#plot
#subplot2015
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [6, 10]})
plt.subplot(2, 1, 1)
ax1 = value_counts_2014.nlargest(15).plot.bar(color = "#CF5D6A", alpha = 0.6)
ax1.yaxis.set_major_locator(MultipleLocator(100))
plt.title("Top 15 items by  purchase frequency \n2014")
plt.tight_layout()
plt.ylabel("Items")
#subplot2014
plt.subplot(2, 1, 2)
ax2 = value_counts_2015.nlargest(15).plot.bar(color = "#082136", alpha = 0.6)
ax2.yaxis.set_major_locator(MultipleLocator(100))
plt.ylabel("Items")
plt.title("2015")
plt.tight_layout()
plt.show()

###Plot Cgnage in items' purchase frequensy from 2014 to 2015
#data transformation
value_counts_2014 = pd.DataFrame(value_counts_2014)
value_counts_2015 = pd.DataFrame(value_counts_2015)
value_counts = value_counts_2014.merge(value_counts_2015, 
                                       left_index=True, right_index=True, 
                                       how = 'left' ,indicator=False)
value_counts["Chnage from 2014 to 2015, %"] = value_counts.apply(lambda row: (row.itemDescription_y / row.itemDescription_x)*100, axis=1)
value_counts["Change from 2014 to 2015, %v2"] = value_counts.apply(lambda row: (row.itemDescription_y / row.itemDescription_x)*100 - 100, axis=1)
value_counts["Chnage from 2014 to 2015"] = value_counts.apply(lambda row: row.itemDescription_y - row.itemDescription_x, axis=1)
value_counts_sorted_perc = value_counts.sort_values("Change from 2014 to 2015, %v2", ascending = False).iloc[:-3,3]
#plot1, increase
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [10, 5]})
ax = value_counts_sorted_perc[:15].plot.bar(color = "#082136", alpha = 0.9)
for p in ax.patches:
        #percentage = '({:.1f}%)'.format(100 * p.get_height()/total)
        ax.annotate('+{:.2f}%'.format(p.get_height()), (p.get_x()-0.15, p.get_height()+5), size=10, color = "k", rotation = 20)
plt.title("Change in items' purchase frequency from 2014 to 2015, %\nTop 15 by increase")
plt.ylabel("Increase, %")
plt.annotate(f"Average change = {value_counts_sorted_perc.mean():.2f}%", xy = (10, 505))
plt.show()
#plot2, decrease
ax2 = value_counts_sorted_perc[len(value_counts_sorted_perc)-15:].sort_values(ascending = True).plot.bar(color = "#082136", alpha = 0.9)
plt.ylim([0, value_counts_sorted_perc[len(value_counts_sorted_perc)-15:].sort_values(ascending = True).min()-10])
for p in ax2.patches:
        ax2.annotate('-{:.2f}%'.format(p.get_height()), (p.get_x()-0.25, p.get_height()), size=10, color = "k", rotation = 20)
plt.title("Change in items' purchase frequency from 2014 to 2015, %\nTop 15 by decrease")
ax2.yaxis.set_major_locator(MultipleLocator(50))
plt.ylabel("Decrease, %")
plt.annotate(f"Average change = {value_counts_sorted_perc.mean():.2f}%", xy = (10, -90))
plt.show()

#set of items
print(list(value_counts_sorted_perc[:15].index)) #incr
print(list(value_counts_sorted_perc[len(value_counts_sorted_perc)-15:].sort_values(ascending = True).index)) #decr

###Adjusted changes
###Increase
#data transformation
adj_change = value_counts.iloc[:,:2]
adj_change = adj_change.dropna()
adj_change_aboveavg = value_counts[value_counts.iloc[:,1]>value_counts.iloc[:,1].mean()]
adj_change_aboveavg = adj_change_aboveavg.sort_values("Chnage from 2014 to 2015", ascending = False)
#plot, increase
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [10, 5]})
ax = adj_change_aboveavg.iloc[:15,4].plot.bar(color = "#082136", alpha = 0.9)
for p in ax.patches:
    #percentage = '({:.1f}%)'.format(100 * p.get_height()/total)
    if p.get_height() > 0:
        ax.annotate('+ {}'.format(p.get_height()), (p.get_x()-0.05, p.get_height()+5), 
                    size=9, color = "k")
    else:
        ax.annotate('- {}'.format(p.get_height()), (p.get_x()-0.15, p.get_height()+0.2), 
                    size=9, color = "k")
plt.title("Adjusted increase in items' purchase frequency from 2014 to 2015, absolute values\nTop 15 by increase")
plt.ylabel("Increase, absolute values")
plt.show()

#data transformation
adj_change_aboveavg = adj_change_aboveavg.sort_values("Change from 2014 to 2015, %v2", ascending = False)
#plot, increase
ax = adj_change_aboveavg.iloc[:15,3].plot.bar(color = "#082136", alpha = 0.9)
for p in ax.patches:
    #percentage = '({:.1f}%)'.format(100 * p.get_height()/total)
    if p.get_height() > 0:
        ax.annotate('+ {:.1f}%'.format(p.get_height()), (p.get_x()-0.15, p.get_height()+5), 
                    size=9, color = "k")
    else:
        ax.annotate('-{:.1f}%'.format(p.get_height()), (p.get_x()-0.15, p.get_height()+0.2), 
                    size=9, color = "k")
plt.title("Adjusted increase in items' purchase frequency from 2014 to 2015, %\nTop 15 by increase")
plt.ylabel("Increase, %")
plt.show()

###Decrease
adj_change_aboveavg = adj_change_aboveavg.sort_values("Chnage from 2014 to 2015", ascending = True)
#plot, increase
ax = adj_change_aboveavg.iloc[:15,4].plot.bar(color = "#082136", alpha = 0.9)
for p in ax.patches:
        ax.annotate('{}'.format(p.get_height()), (p.get_x(), p.get_height()-5), 
                    size=9, color = "k")
plt.title("Adjusted decrese in items' purchase frequency from 2014 to 2015, absolute values\nTop 15 by decrease")
plt.ylabel("Decrese, absolute values")
plt.show()

#data transformation
adj_change_aboveavg = adj_change_aboveavg.sort_values("Change from 2014 to 2015, %v2", ascending = True)
#plot, decrese
ax = adj_change_aboveavg.iloc[:15,3].plot.bar(color = "#082136", alpha = 0.9)
for p in ax.patches:
    #percentage = '({:.1f}%)'.format(100 * p.get_height()/total)
    if p.get_height() > 0:
        ax.annotate('+ {:.1f}%'.format(p.get_height()), (p.get_x()-0.15, p.get_height()-5), 
                    size=9, color = "k")
    else:
        ax.annotate('-{:.1f}%'.format(p.get_height()), (p.get_x()-0.15, p.get_height()-1.5), 
                    size=9, color = "k")
plt.title("Adjusted decrese in items' purchase frequency from 2014 to 2015, %\nTop 15 by decrease")
plt.ylabel("Decrese, %")
plt.show()

######### Member number
df.Member_number.nunique() #3898
df_reset_index = df.reset_index()
#description
import sidetable as stb
#count of customers per year
df_reset_index.groupby("Year").agg(dict(Member_number = "count")).stb.subtotal() 
#unique customers per year
df_reset_index.groupby("Year").agg(dict(Member_number = "nunique")).stb.subtotal() 

###Plot 30 customers by purchase amount
#data transformation
grouped_be_year_purc = df_reset_index.groupby("Member_number").agg(Purchased = ("itemDescription", "count"))
grouped_be_year_purc = grouped_be_year_purc.sort_values("Purchased", ascending = False)
#plot
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [10, 3]})
ax = grouped_be_year_purc.iloc[:30,:].plot.bar(legend = False, alpha = 0.75)
plt.title("Top 30 customers by purchase amount")
plt.annotate(f"Average = {grouped_be_year_purc.mean()[0]:.2f}", xy = (25,35), fontsize = 12)
plt.xlabel("Member number")
plt.ylabel("Purchase amount")
plt.show()

###Plot unique custmers per week
#data transformation
grouped_by_w_y = df.groupby(["Weekday", "Year"], as_index = False).agg(Customer_n = ("Member_number", "nunique"))
grouped_by_w_y["Weekday"] =  pd.Categorical(grouped_by_w_y["Weekday"], ['Monday', 'Tuesday', 'Wednesday', "Thursday", "Friday", "Saturday", "Sunday"])
grouped_by_w_y.sort_values("Weekday")
#plot
sns.set(style = "whitegrid", 
        palette= "rocket", 
        font_scale = 1,
        rc={"figure.figsize": [8, 3]})
ax = sns.lineplot(x = grouped_by_w_y["Weekday"], y = grouped_by_w_y["Customer_n"], hue = grouped_by_w_y["Year"])
plt.title("Amount of unique customers per weekday, \n2014 vs 2015")
ax.yaxis.set_major_locator(MultipleLocator(50))
grouped_by_w_y["Year"] = grouped_by_w_y["Year"].astype("string")
mean2014 = grouped_by_w_y[grouped_by_w_y["Year"] == "2014"]["Customer_n"].mean()
mean2015 = grouped_by_w_y[grouped_by_w_y["Year"] == "2015"]["Customer_n"].mean()
plt.axhline(mean2014, linewidth = 1, alpha = 0.7, linestyle = "--", color = "k")
plt.axhline(mean2015, linewidth = 1, alpha = 0.7, linestyle = "--", color = "k")
plt.annotate(f"Average:\n 2014: {mean2014:.2f}\n 2015: {mean2015:.2f}", xy = (5.2, 930), fontsize = 10)
plt.ylabel("Number of customers")
plt.legend(loc = (0,1))
plt.xlim(0,6)
plt.show()

###Plot amount of purchases for unique custmers per week
#data transformation
grouped_by_w_y_and_item_count = df.groupby(["Weekday", "Year"], as_index = False).agg(Customer_n = ("Member_number", "nunique"), Purchases = ("itemDescription", "count"))
grouped_by_w_y_and_item_count["Weekday"] =  pd.Categorical(grouped_by_w_y_and_item_count["Weekday"], ['Monday', 'Tuesday', 'Wednesday', "Thursday", "Friday", "Saturday", "Sunday"])
grouped_by_w_y_and_item_count.sort_values("Weekday")
grouped_by_w_y_and_item_count["Purchases_per_customer"] = grouped_by_w_y_and_item_count.apply(lambda row: row.Purchases / row.Customer_n, axis=1)
#plot
ax = sns.lineplot(x = grouped_by_w_y_and_item_count["Weekday"], y = grouped_by_w_y_and_item_count["Purchases_per_customer"], hue = grouped_by_w_y["Year"])
plt.title("Amount of purchases by unique customers per weekday, \n2014 vs 2015")
ax.yaxis.set_major_locator(MultipleLocator(0.1))
grouped_by_w_y_and_item_count["Year"] = grouped_by_w_y_and_item_count["Year"].astype("string")
mean2014_per_customer = grouped_by_w_y_and_item_count[grouped_by_w_y_and_item_count["Year"] == "2014"]["Purchases_per_customer"].mean()
mean2015_per_customer = grouped_by_w_y_and_item_count[grouped_by_w_y_and_item_count["Year"] == "2015"]["Purchases_per_customer"].mean()
plt.axhline(mean2014_per_customer, linewidth = 1, alpha = 0.3, linestyle = "--", color = "k")
plt.axhline(mean2015_per_customer, linewidth = 1, alpha = 0.3, linestyle = "--", color = "k")
plt.annotate(f"Average:\n 2014: {mean2014_per_customer:.2f}\n 2015: {mean2015_per_customer:.2f}", xy = (5.4, 2.9), fontsize = 10)
plt.ylabel("Amount of purchses")
plt.legend(loc = (0.885,1))
plt.xlim(0,6)
plt.show()

