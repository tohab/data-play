#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load CSV file (with a header)
    df = pd.read_csv('data.csv')
    
    # Convert column O (15th column, index 14) to datetime.
    df['date'] = pd.to_datetime(df.iloc[:, 14])
    
    # Each row represents an email sent.
    df['sent'] = 1
    
    # Column M (13th column, index 12) indicates a response if not blank.
    df['responded'] = df.iloc[:, 12].notna().astype(int)
    
    # Group by date to get daily totals and sort by date.
    daily = df.groupby('date').agg({'sent': 'sum', 'responded': 'sum'}).sort_index()
    
    # Compute the number of emails that did not get a response.
    daily['not_responded'] = daily['sent'] - daily['responded']
    
    # Calculate cumulative totals using numpy.
    daily['cum_sent'] = np.cumsum(daily['sent'])
    daily['cum_responded'] = np.cumsum(daily['responded'])
    daily['cum_not_responded'] = daily['cum_sent'] - daily['cum_responded']
    
    # Create a stacked area chart with cumulative responded emails on the bottom.
    plt.figure(figsize=(10, 6))
    plt.stackplot(daily.index,
                  daily['cum_responded'],       # Base: cumulative responded
                  daily['cum_not_responded'],   # Top: cumulative not responded
                  labels=['Cumulative Responded', 'Cumulative Not Responded'],
                  alpha=0.8)
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Emails Sent')
    plt.title('Cumulative Emails Sent and Responses Over Time')
    plt.xticks(rotation=45)
    
    # Set the x-axis from the earliest date in the data to the current date.
    plt.xlim([daily.index.min(), pd.Timestamp.today()])
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('emails_plot.png')

if __name__ == '__main__':
    main()
