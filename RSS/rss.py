import feedparser as fp
import os
import pandas as pd
from datetime import datetime

# Setting up feeds as global variable
iTunes = "http://ax.itunes.apple.com/WebObjects/MZStoreServices.woa/ws/RSS/topMovies/xml"
BBC = "http://feeds.bbci.co.uk/news/world/rss.xml"
rss_list = [iTunes, BBC]

# Setting up dataframe columns as
df_columns = ['sourse', 'date', 'title', 'link', 'summary']

# Function to parse data and append to DF
def RSS(rss, df):
    feed = fp.parse(rss)

    if feed.bozo == False:
        print('%s is well formed feed..!!' % feed.feed.title)
    else:
        print('%s has flipped the bozo bit, potential risk ahead..!!' %
              feed.feed.title)

    # Set feed time to published date if found, else set the current date
    feed_date = feed.feed.get('updated', datetime.now().strftime('%Y-%m-%d'))

    # Counter for loop
    i = 0

    # Append required information to dataframe
    while i < 10:
        feed_item = pd.Series([feed.feed.title, feed_date, feed.entries[i].title,
                               feed.entries[i].id, feed.entries[i].summary], df_columns)
        df = df.append(feed_item, ignore_index=True)
        i += 1

    # Return dataframe
    return df


if __name__ == "__main__":

    # Create empty dataframe
    _df = pd.DataFrame(columns=df_columns)

    for item in rss_list:
        _df = RSS(item, _df)

    # Save to csv, if csv already exist, append to it
    if not os.path.isfile('RSS.csv'):
        _df.to_csv('RSS.csv', header=df_columns, index=False)
    else:
        _df.to_csv('RSS.csv', mode='a', header=False, index=False)
