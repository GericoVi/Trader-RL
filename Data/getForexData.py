##################################
## Scrape forex tick data
##################################

import datetime as dt
import numpy as np
import pandas as pd
import requests
import os
import gzip
from io import StringIO
from fxcmpy import fxcmpy_tick_data_reader as tdr
from tqdm import tqdm

# Download tick data from tickdata.fxcorporate.com as .gz files
def scrapeForexTickData(years,symbols=[], outputDir='gz'):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # By default will try and extract all available symbols for specified years
    if not symbols:
        symbols = tdr.get_available_symbols()

    # Construct the years, weeks and symbol lists required for the scraper.
    weeks = list(range(1,53))

    for pair in symbols:
        if pair not in tdr.get_available_symbols():
            raise Exception(f"Currency pair {pair} not available in fxcmpy api")

    # Put each pairs data into separate directory
    for symbol in symbols:    
        print(f"Downloading data for {symbol}...")
        if not os.path.exists(f"{outputDir}/{symbol}"):
            os.makedirs(f"{outputDir}/{symbol}")

        # Scrape available data and use a progress bar
        with tqdm(total=len(years)*len(weeks), unit='file') as pbar:
            for year in years:
                for week in weeks:
                    url = f"https://tickdata.fxcorporate.com/{symbol}/{year}/{week}.csv.gz"
                    r = requests.get(url, stream=True)
                    with open(f"{outputDir}/{symbol}/{symbol}_{year}_w{week}.csv.gz", 'wb') as file:
                        for chunk in r.iter_content(chunk_size=1024):
                            file.write(chunk)
                    pbar.update(1)

    # Check all the files for each currency pair was downloaded (should be 104 for each)
    total = 0
    print()
    print('Checking...')
    for symbol in symbols:
        count = 0
        for file in os.listdir(f"{outputDir}/{symbol}"):
            if file[:6] == symbol:
                count+=1
        total += count
        print(f"{symbol} files downloaded = {count} ")
    print(f"\nTotal files downloaded = {total}")

# Extract, transform and load the .gz files to monthly csvs - for reading with dask
def ETLtoCSV(pairs, gzDirectory='gz', outputDir='csv'):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Check input
    if not isinstance(pairs, list):
        raise Exception("Input should be a list")

    # Loop through specified currency pairs and all .gz files within
    for pair in pairs:
        print(f"Extracting files in {pair} directory")
        if not os.path.exists(f"{outputDir}/{pair}"):
            os.makedirs(f"{outputDir}/{pair}")

        for file in tqdm(os.listdir(f"{gzDirectory}/{pair}"), unit='file'):
            if file.endswith('.gz'):
                #print(f"\nExtracting: {file}")

                # extract gzip file and assign to Dataframe
                codec = 'utf-16'
                f = gzip.GzipFile(f'{gzDirectory}/{pair}/{file}')
                try:
                    data = f.read()
                    data_str = data.decode(codec)
                    df = pd.read_csv(StringIO(data_str))
                    
                    # Pad missing zeroes in microsecond field
                    df['DateTime'] = df.DateTime.str.pad(26, side='right', fillchar='0')
                    
                    # Find out format of date time within this file and convert to actual date time object
                    #### Faster to do this way than let pandas infer the format - but there may be a better algorithm than this
                    sampleStart = df['DateTime'][0]
                    sampleEnd = df['DateTime'][len(df)-1]
                    if (sampleStart[2] != '/' and sampleStart[2] != '-'): # Means year is first
                        separator = sampleStart[4]
                        # For any given week, change in day number will be larger than change in month number
                        dayMonth = abs(int(sampleStart[5:7])-int(sampleEnd[5:7])) > abs(int(sampleStart[8:10])-int(sampleEnd[8:10]))
                        if dayMonth:
                            df['DateTime'] = pd.to_datetime(df['DateTime'], format = f'%Y{separator}%d{separator}%m %H:%M:%S.%f')
                        else:
                            df['DateTime'] = pd.to_datetime(df['DateTime'], format = f'%Y{separator}%m{separator}%d %H:%M:%S.%f')
                    else:
                        separator = sampleStart[2]
                        # For any given week, change in day number will be larger than change in month number
                        dayMonth = abs(int(sampleStart[0:2])-int(sampleEnd[0:2])) > abs(int(sampleStart[3:5])-int(sampleEnd[3:5]))
                        if dayMonth:
                            df['DateTime'] = pd.to_datetime(df['DateTime'], format = f'%d{separator}%m{separator}%Y %H:%M:%S.%f')
                        else:
                            df['DateTime'] = pd.to_datetime(df['DateTime'], format = f'%m{separator}%d{separator}%Y %H:%M:%S.%f')

                    # Make sure there's no unnecessary trailing zeros on the tick data
                    df['Bid'] = [ float('%.6g' % tick) for tick in df['Bid'] ]
                    df['Ask'] = [ float('%.6g' % tick) for tick in df['Ask'] ]

                    # Assign Datetime column as index so that it doesn't write a meaningless index to csv 
                    df.set_index('DateTime', inplace=True)

                    # Write data to csv
                    df.to_csv(f"{outputDir}/{pair}/{file[:-3]}")

                # Catch error since scrape function will create a bad gzip file if the file it requests is not there at the url
                except gzip.BadGzipFile:
                    print(f"\nBad gzip file, skipping {file}")

# Process ticks together into bins (hourly or daily)
def processticks(pairs, years, weeks=list(range(1,53)), bin='1H', ticksDir='csv', outputDir='data'):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        
    # Check inputs
    if not isinstance(pairs, list):
        raise Exception("Input 'pairs' should be a list")
    if not isinstance(years, list):
        raise Exception("Input 'years' should be a list")
    if not isinstance(weeks, list):
        raise Exception("Input 'weeks' should be a list")

    # Loop through all requested files
    for pair in pairs:
        # Make directory for forex pair if not there
        if not os.path.exists(f"{outputDir}/{pair}"):
                os.makedirs(f"{outputDir}/{pair}")

        # Make directory for specific bin type
        if not os.path.exists(f"{outputDir}/{pair}/{bin}"):
                os.makedirs(f"{outputDir}/{pair}/{bin}")

        print(f'Processing {pair} directory for {bin} bins')

        # Initialise progress bar
        with tqdm(total=len(years)*len(weeks), unit='file') as pbar:
            for year in years:
                for week in weeks:
                    try:
                        # Read in as simple pandas dataframe
                        df = pd.read_csv(f'{ticksDir}/{pair}/{pair}_{year}_w{week}.csv')

                        # Convert to actual datetime so can mask - specify format, much faster as pandas doesn't have to infer
                        df['DateTime'] = pd.to_datetime(df['DateTime'], format = '%Y-%m-%d %H:%M:%S')

                        # Make bin specific variables for creating mask later and stop
                        if bin == '1H':
                            timeIncrement = dt.timedelta(hours = 1)
                            # Rounded first time for anchor
                            firstTime = df['DateTime'][0].replace(minute=0, second=0, microsecond=0)
                            # Get stop time
                            lastTimeTrunc = df['DateTime'][len(df)-1].replace(minute=0, second=0, microsecond=0) + timeIncrement

                        elif bin == '4H':
                            timeIncrement = dt.timedelta(hours = 4)
                            # Rounded first time for anchor
                            firstTime = df['DateTime'][0].replace(minute=0, second=0, microsecond=0)
                            firstTime = firstTime + dt.timedelta(hours = (int(firstTime.hour/4)*4) - firstTime.hour) # There's probably a better way of doing this - 'rounding to floor, multiple of 4'
                            # Get stop time
                            lastTimeTrunc = df['DateTime'][len(df)-1].replace(minute=0, second=0, microsecond=0)
                            lastTimeTrunc = lastTimeTrunc + dt.timedelta(hours = (int(lastTimeTrunc.hour/4)*4) - lastTimeTrunc.hour)
                        
                        elif bin == '1D':
                            timeIncrement = dt.timedelta(days = 1)
                            # Rounded first time for anchor
                            firstTime = df['DateTime'][0].replace(hours=0,minute=0, second=0, microsecond=0)
                            # Get stop time
                            lastTimeTrunc = df['DateTime'][len(df)-1].replace(hours=0,minute=0, second=0, microsecond=0) + timeIncrement

                        else:
                            raise Exception(f'{bin} bin not implemented')

                        # Initialise new dataframe
                        index = pd.date_range(start=firstTime, end=lastTimeTrunc, freq = bin)
                        columns = ['B_Open','B_High','B_Low','B_Close','A_Open','A_High','A_Low','A_Close']
                        newdf = pd.DataFrame(index=index, columns=columns)
                        newdf.index.name = 'DateTime'

                        # Step along rows group them according to bins
                        startTime = firstTime
                        while startTime != lastTimeTrunc:
                            mask = (df['DateTime'] > startTime) & (df['DateTime'] < startTime+timeIncrement)
                            subset = df.loc[mask].reset_index()
                            # Get stats for bid and ask (if there is tick data for that hour)
                            if (len(subset) > 1):
                                newdf.loc[startTime] = {'B_Open': float('%.5g' % subset['Bid'][0]), 
                                                        'B_High': float('%.5g' % subset['Bid'].max()), 
                                                        'B_Low': float('%.5g' % subset['Bid'].min()), 
                                                        'B_Close': float('%.5g' % subset['Bid'][len(subset)-1]),
                                                        'A_Open': float('%.5g' % subset['Ask'][0]), 
                                                        'A_High': float('%.5g' % subset['Ask'].max()), 
                                                        'A_Low': float('%.5g' % subset['Ask'].min()), 
                                                        'A_Close': float('%.5g' % subset['Ask'][len(subset)-1])    }
                            elif (len(subset) == 1):
                                bidVal = float('%.6g' % subset['Bid'][0])
                                askVal = float('%.6g' % subset['Ask'][0])
                                newdf.loc[startTime] = {'B_Open': bidVal, 'B_High': bidVal, 'B_Low': bidVal, 'B_Close': bidVal,
                                                        'A_Open': askVal, 'A_High': askVal, 'A_Low': askVal, 'A_Close': askVal    }
                            else:
                                newdf.drop([startTime], inplace = True)

                            # Move on
                            startTime = startTime + timeIncrement

                        # Save new dataframe
                        newdf.to_csv(f'{outputDir}/{pair}/{bin}/{year}_w{week}.csv')

                        pbar.update(1)
                    except FileNotFoundError:
                        #print(f'\n{ticksDir}/{pair}/{pair}_{year}_w{week}.csv not found, SKIPPING')
                        pbar.update(1)
              
#ETLtoCSV(['EURUSD', 'EURGBP', 'USDJPY', 'GBPUSD'])

#processticks(['EURUSD', 'EURGBP', 'USDJPY', 'GBPUSD'],[2018,2019,2020])

#symbols = tdr.get_available_symbols()
#symbols = symbols[3:]
#scrapeForexTickData([2018, 2019, 2020], symbols)