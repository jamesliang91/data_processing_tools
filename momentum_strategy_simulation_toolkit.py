import os
import numpy as np
import pandas as pd
import datetime as dt
import time
import matplotlib.pyplot as plt
import logging
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import autocorrelation_plot
logging.basicConfig()
logger = logging.getLogger()


"""
Function 1:
read all the file object (ignore non-file object) under a given directory, check for data file's structure (column number) and read all valid stock_bar_data file as csv. Output a dataframe with all data concatenated from seperate files.
expected run time: < 10s

"""


def read_stock_time_bar_to_csv(target_dir, num_of_cols=10):
        
    """
        Arguments:
            target_dir -- the directory that contains stock bar data OR the filepath of a data file
            num_of_cols -- number of columns expected from market data file, used to filter out data files 
                         without correct structure
            
        Returns:
            *abnormal(for debugging) -- the data file that is not integrated into output_df
            output_df -- stock bar data being integrated
    
    """
    
    file_path_list = []
    
    if os.path.isfile(target_dir) == True:
        file_path_list.append(target_dir)
    elif os.path.isdir(target_dir) == True:
        for file in os.listdir(target_dir):
            # Note: add [if not file.startswith('.'):] if reading process captured file such as DS_store.
            file_path_list.append(os.path.join(target_dir,file))
    else:
        return ("input is neither a filepath nor a directory, please check.")
    # create a list to store df objects 
    data_list = []
    
    # if file consist of abnormal data (i.e. number of columns is wrong), will be append here
    abnormal_file_list = []
    
    for item in file_path_list:
        if os.path.isfile(item) == True:
            # Note: the Instrument ID need to be read as string instead of int, 
            #otherwise id such as 000034 will be convert to 34!!!
            
            df = pd.read_csv(item, index_col=None, header=0, dtype={"bar_InstrumentID": str})
            if len(df.columns) != num_of_cols:
                abnormal_file_list.append(item)
                continue
            else:
                data_list.append(df)
    
    if len(abnormal_file_list) > 0:
        print ("Abnormal filepath or Non-file path detected:")
        return ("\n".join(abnormal_file_list))
        
    output_df = pd.concat(data_list, axis=0, ignore_index = True)
        
    return (output_df)






"""
Function 2:
for a given dataframe which consist of stock bar data, Calculate stock_time_bar return
expected run time: <1s

"""


def calculate_stock_return(data_df):
    """
        Arguments:
            data_df: input data frame, expect to have the following attribute:
                     bar_Timestamp, bar_ExchangeID, bar_InstrumentID, bar_Open_Price, bar_Close_Price, 
                     bar_High_Price, bar_Low_Price, bar_BidAskMean, bar_Cumulative_Volume, bar_Volume
            
        Returns:
            data_df: with appended columns bar_Return and asset_ID; rows reordered by asset_ID
            
    """
    
    data_df["bar_Return"] = data_df["bar_Close_Price"]/data_df["bar_Open_Price"]
    data_df["asset_ID"] = data_df["bar_ExchangeID"].astype(str) + data_df["bar_InstrumentID"].astype(str)
    data_df = data_df.reindex(columns=["bar_Timestamp","asset_ID","bar_ExchangeID","bar_InstrumentID","bar_Open_Price",
                                       "bar_Close_Price","bar_High_Price","bar_Low_Price","bar_BidAskMean",
                                       "bar_Cumulative_Volume","bar_Volume","bar_Return"])
    data_df.sort_values(by=["asset_ID"])
    data_df.reset_index(drop=True)
    
    return data_df






"""
Function 3:
momentum strategy simulator tool 1: momentum portfolio constructor
portfolios are constructed based on the cumulative return
expected run time: ~40s

"""


def momentum_portfolio_constructor(data_df, time_point, lookback_period, lookforward_period, lag_period = 0):
    
    """
        Arguments:
            data_df -- input data frame, expect to have the following attribute:
                   bar_Timestamp, bar_ExchangeID, bar_InstrumentID, bar_Open_Price, bar_Close_Price, 
                   bar_High_Price, bar_Low_Price, bar_BidAskMean, bar_Cumulative_Volume, bar_Volume
            time_point -- the timepoint to simulate momentum strategy (i.e. construct portfolio)
            lookback_period -- the length of period (in past) we choose to examine the return of assets, 
                         such an examination will determine the "winner" and "loser" portfolio
            lookforward_period -- the length of period (in future) we choose to maintain our long-short 
                           position on "winner" and "loser" portfolio
            lag_period -- the length of period that we "wait" before we form the long-short position based 
                      on our momentum strategy at a particular point in time (e.g.: a lag period of 1 
                      means that at timepoint t, after we have calculated all stock's return over the 
                      look-back period and constructed momentum winner & loser portfolios, we execute 
                      our strategy (buy winner, sell loser, maintain position for look-forward period
                      and then clear those positions) at timepoint t+1), Default is 0
                                    
        Returns:
            stock_rank_df: the ranked stock return dataframe (ranked on lookback_period return), by asset
            incomplete_asset_dict: for debugg, consist of assetID and error details for asset that failed to
                           during return calculations
    
    """  
    # step 1: for each asset, locate time point to start, check if there are enough data in files that satisfies look-back
    # period; if not, use 1 as return rate to fill in missing time intervals when we compute the compound return of an asset
    
    # initialize stock rank dataframe, stock_list (by asset ID) 
    header = list(data_df.columns.values)
    header.append("lookback_cumulative_return") # the cumulative return over the lookback period
    header.append("lookback_close") # the close price at the end of the lookback period
    header.append("lag_period")  # the lag period before portfolio formation
    header.append("lag_bid_ask_mean") # the close price at the end of the lag period (= close price at time_point if lag is 0)
    header.append("lookforward_cumulative_return") # the cumulative return over the lookforward period (with lag)
    header.append("lookforward_bid_ask_mean") # the close price at the end of the lookforward period (with lag)
    
    stock_rank_df = pd.DataFrame(columns = header)
    
    # stock_rank_df.insert(len(data_df.columns),"lookback_cumulative_return","")
    asset_list = list(data_df["asset_ID"].unique())
    incomplete_asset_dict = {}
    
    # initialize timestamp object (to 19-digits, cater to the format of bar_Timestamp)
    timestamp = int(time_point * (10 ** 9))
    
    # for each asset, calculate cumulative return over look-back period, then append result data into dataframe 
    for asset in asset_list:
        # initialize information to be returned
        asset_return_info = []
        error_info = []
        
        # get corresponding asset's information from data_df
        asset_info = data_df.loc[data_df["asset_ID"]==asset]
        asset_info = asset_info.sort_values(by=["bar_Timestamp"]).reset_index(drop=True)
        
        # retrieve the index of the row with same or closest timestamp from time_point ( Note: most of the time we
        # can not find a row with precise timestamp as time_point, especially when we target on a group of asset)
        # here we find such an index by subtracting the time from bar_Timestamp, take absolute value and find the 
        # row with smallest absolute value (closest)
        
        start_index = asset_info["bar_Timestamp"].sub(timestamp).abs().idxmin()
        asset_max_index = max(list(asset_info.index))
        
        # append the asset's information at the start_index
        asset_return_info.extend(asset_info.loc[start_index].values)
        
        # calculate lookback period cumulative return using lookback close price; if not enough data, use the closest 
        lookback_close = asset_info["bar_Close_Price"].loc[max(start_index-lookback_period,0)]
        if lookback_close == 0:
            error_info.append("close price at lookback timepoint is zero")
            
        elif max(start_index-lookback_period,0) == 0:
            error_info.append("data not enough to cover lookback period from given start timepoint")
            
        else:
            lookback_cumulative_return =  asset_info["bar_Close_Price"].loc[start_index] / lookback_close
        
        # acquire the lag period close
        lag_bid_ask_mean = asset_info["bar_BidAskMean"].loc[min(start_index+lag_period, asset_max_index)]
        if lag_bid_ask_mean == 0:
            error_info.append("bid ask mean price at lag time point is zero")
            
        # calculate lookforward period cumulative return using lookforward close price; if not enough data, use the closest 
        lookforward_bid_ask_mean = asset_info["bar_BidAskMean"].loc[min(start_index+lag_period+lookforward_period, asset_max_index)]
        if lookforward_bid_ask_mean == 0:
            error_info.append("close price at lookforward period is zero")
            
        elif min(start_index+lag_period+lookforward_period, asset_max_index) == asset_max_index:
            error_info.append("data not enough to cover lookback period from given start timepoint")
            
        else:
            lookforward_cumulative_return =  lookforward_bid_ask_mean / lag_bid_ask_mean
        
        if len(error_info) >= 1:
            incomplete_asset_dict[asset] = error_info
            continue
        
        # append lookback & lag & lookforward info to dataframe
        asset_return_info.extend([lookback_cumulative_return, lookback_close, lag_period,
                                 lag_bid_ask_mean, lookforward_cumulative_return, lookforward_bid_ask_mean])
        
        asset_return_info = pd.Series(asset_return_info, index = stock_rank_df.columns)
        stock_rank_df = stock_rank_df.append(asset_return_info,ignore_index = True)
    
    # Reorder the stock_rank_df by lookback cumulative return (for momentum winner & loser portfolio constructions later)
    stock_rank_df = stock_rank_df.sort_values(by=["lookback_cumulative_return"],ascending = False).reset_index(drop=True)
    
    return stock_rank_df, incomplete_asset_dict
    

    
    
    

"""
Function 4:
momentum strategy simulator tool 2: momentum strategy return calculator
calculate the cumulative returns of momentum winner & loser
expected run time: <1s

"""


def momentum_strategy_return_calculator (stock_rank_df, percentile_rate = 5, fee_rate = 0.0007):
    
    """
        Arguments:
            data_df -- the processed stock_bar_data dataframe object (with return value) that we want to research on
                       Note that this dataframe need to be ranked on lookback return rate with descending order
            percentile_rate -- int, percentile used to perform asset segmentation (by return) and portfolio formation
                               (winner&loser) recommended to be 5-10, default to be 5 (quintile portfolio)
            fee_rate -- transaction fee rate, default to be 0.0007; here we use 0.07% as Chinese A share market transaction fee
            
        Returns:
            stock_rank_df: a dataframe consist of important information for momentum portfolio construction and trading
    
    """     

    row_num = stock_rank_df.shape[0]
    portfolio_index = int(row_num * (1/percentile_rate))
    winner_portfolio = stock_rank_df.iloc[0:portfolio_index+1]
    loser_portfolio = stock_rank_df.iloc[-(portfolio_index+1):-1]
    
    # calculate winner portfolio return rate (long position)
    winner_portfolio_initial_pos = np.sum(np.multiply(winner_portfolio["lag_bid_ask_mean"],100))
    winner_portfolio_end_pos = np.sum(np.multiply(winner_portfolio["lookforward_bid_ask_mean"],100))
    winner_return = winner_portfolio_end_pos - winner_portfolio_initial_pos
    winner_fee =  (winner_portfolio_initial_pos + winner_portfolio_end_pos) * fee_rate
    
    # calculate loser portfolio return rate (short position)
    loser_portfolio_initial_pos = np.sum(np.multiply(loser_portfolio["lag_bid_ask_mean"],100))
    loser_portfolio_end_pos = np.sum(np.multiply(loser_portfolio["lookforward_bid_ask_mean"],100)) 
    loser_return = loser_portfolio_initial_pos - loser_portfolio_end_pos
    loser_fee = (loser_portfolio_initial_pos + loser_portfolio_end_pos) * fee_rate
    
    #return winner_return_rate, loser_return_rate
    return winner_portfolio_initial_pos, winner_portfolio_end_pos, winner_fee,\
         loser_portfolio_initial_pos, loser_portfolio_end_pos, loser_fee





"""
Function 5:
momentum strategy simulator tool 3: momentum strategy simulator
use momentum portfolio constructor and return calculator to simulate momentum strategy multiple
times during a period (i.e. on multiple timepoint within the period)
expected run time: 300-500s 

"""



def momentum_strategy_simulator(data_df, start_time, end_time, time_interval,
                                lookback_period, lookforward_period, lag_period = 0,
                                percentile_rate = 5, fee_rate = 0.0007):
    
    """
        Arguments:
            data_df -- the processed stock_bar_data dataframe object (with return value) that we want to research on
            start_time -- the starting timepoint to simulate momentum strategy in a day -- a unix 10-digits timestamp
            end_time -- the ending timepoint to simulate momentum strategy in a day -- a unix 10-digits timestamp
            time_interval -- time interval of the data_df (consistent with the time interval of bar data) in seconds
            lookback_period -- int, length of period used in momentum portfolio formation
            lookforward_period -- int, length of period used in momentum 
            lag_period -- int, length of period that we "skip" before portfolio formation, default to be 0
            percentile_rate -- int, percentile used to perform asset segmentation (by return) and portfolio formation
                               (winner&loser) recommended to be 5-10, default to be 5 (quintile portfolio)
            fee_rate -- transaction fee rate, default to be 0.0007; here we use 0.07% as Chinese A share market transaction fee
            
        Returns:
            momentum_portfolio_return_dict -- a dictionary consist of returns of winner & loser portfolios constructed
                                              on every pre-defined timepoint 
            cumulative_daily_return_dict -- a dictionary consist of 1-day cumulative return of winner & loser portfolio 
    
    """  
    
    market_start_time_stamp = int(data_df.iloc[0]["bar_Timestamp"]*(10**(-9)))
    # Note: 5 hour 20 min (from 9:35 am to 14:55 pm) = 19200 seconds
    market_end_time_stamp = data_df.iloc[0]["bar_Timestamp"] + 19200
    
    # check if user defined timestamp is out of range (i.e. not enough data for lookback or lookforward period)
    assert (start_time - lookback_period * time_interval) >= market_start_time_stamp,\
    "start time point too early, choose a later timestamp"
    assert (end_time + lookforward_period * time_interval) <= market_end_time_stamp,\
    "end time point too late, choose a ealier timestamp"
    
    # use range function to get a list of timestamps
    ori_time_point_list = list(range(int(start_time), int(end_time)+1, time_interval))
    time_point_list = []
    for i in range(len(ori_time_point_list)):
        
        # check if a given timestamp is in active trading hours (i.e. 9:30am-11:30am; 1:00pm-3:00pm ),use time() function
        # to extract hour information from a datetime object
        str_time = dt.datetime.fromtimestamp(int(ori_time_point_list[i]))
        if str_time.time() < dt.datetime(2021,6,1,9,30).time():
            continue
        elif dt.datetime(2021,6,1,11,30).time() < str_time.time() < dt.datetime(2021,6,1,13,0).time():
            continue
        elif dt.datetime(2021,6,1,14,57).time() < str_time.time():
            continue
        time_point_list.append(int(ori_time_point_list[i]))
    
    
    momentum_portfolio_return_dict = {}
    
    # Generate momentum portfolios on selected timepoints
    for timepoint in time_point_list:
        stock_rank_df, incomplete_asset_dict = momentum_portfolio_constructor(data_df = data_df,
                                                                              time_point = timepoint,
                                                                              lookback_period = lookback_period,
                                                                              lookforward_period = lookforward_period,
                                                                              lag_period = lag_period)
    
        winner_portfolio_initial_pos,\
        winner_portfolio_end_pos,\
        winner_fee,\
        loser_portfolio_initial_pos,\
        loser_portfolio_end_pos,\
        loser_fee = momentum_strategy_return_calculator(stock_rank_df = stock_rank_df, 
                                                        percentile_rate = percentile_rate,
                                                        fee_rate = fee_rate)
        
        momentum_portfolio_return_dict[str(timepoint)] = [winner_portfolio_initial_pos, winner_portfolio_end_pos, winner_fee,\
                                                          loser_portfolio_initial_pos, loser_portfolio_end_pos, loser_fee]
    
    winner_acc_profit = 0
    loser_acc_profit = 0
    winner_acc_base = 0
    loser_acc_base = 0
    
    # caculate the cumulative return rate (if momentum strategy is executed consectively on every select timepoint)
    # Note: this return rate is used ONLY to verify the existence of P&L margin, in reality a consecutive high-frequency
    # trading scheme is impractical
    
    for item in list(momentum_portfolio_return_dict.values()):
        winner_acc_profit += (item[1]-item[0]-item[2])
        winner_acc_base += item[0]
        loser_acc_profit += (item[3]-item[4]-item[5])
        loser_acc_base += item[3]
    
    cumulative_daily_return_dict = {"daily_winner_acc_return": (winner_acc_profit/winner_acc_base),
                                    "daily_loser_acc_return": (loser_acc_profit/loser_acc_base) }
        
    return momentum_portfolio_return_dict, cumulative_daily_return_dict
    


    
    
    
"""
Function 6:
momentum strategy simulator tool 4: multiday momentum strategy simulator
use momentum strategy simulator to simulate momentum strategy multiple times during a period 
(i.e. on multiple timepoint within the period) on multiple day's data
expected run time: 300-500s * number of data files

"""

def multiday_momentum_strategy_simulator(target_dir, output_file_path, start_time, end_time, time_interval,
                                         lookback_period, lookforward_period, lag_period = 0, percentile_rate = 5,
                                         fee_rate = 0.0007, num_of_cols=10, return_output=True) :
    """
        Arguments:
            target_dir: the directory that contains stock bar data OR the filepath of a data file
            output_file_path: the filepath to store the output data file
            
            start_time -- the starting timepoint to simulate momentum strategy in a day -- datetime object 
            end_time -- the ending timepoint to simulate momentum strategy in a day -- 
            time_interval -- time interval of the data_df (consistent with the time interval of bar data) in seconds
            lookback_period -- int, length of period used in momentum portfolio formation
            lookforward_period -- int, length of period used in momentum 
            lag_period -- int, length of period that we "skip" before portfolio formation, default to be 0
            percentile_rate -- int, percentile used to perform asset segmentation (by return) and portfolio formation 
                               (winner&loser) recommended to be 5-10, default to be 5 (quintile portfolio)
            fee_rate -- transaction fee rate, default to be 0.0007; here we use 0.07% as Chinese A share market transaction fee
            num_of_cols -- number of columns expected from market data file, used to filter out data files without correct 
                           structure
            return_output -- whether to read the output file and final output  , default is True
            
        Returns:
            output_df(optional to user) -- (read) the resulted data file as a dataframe
        
    
    """
    file_path_list = []
    
    # check target directory
    if os.path.isfile(target_dir) == True:
        file_path_list.append(target_dir)
    elif os.path.isdir(target_dir) == True:
        for file in os.listdir(target_dir):
            file_path_list.append(os.path.join(target_dir,file))
    else:
        raise ValueError("input is neither a filepath nor a directory, please check.")
    
    # check output directory
    if not os.path.exists(output_file_path):
        raise ValueError("output_file_path does not exist, please check.")    
    
    today_info = str(dt.date.today())
    output_file_name = "intraday_momentum_strategy" + "_" + str(lookback_period) + "_" + str(lag_period) + "_" + \
                        str(lookforward_period) + "_" + today_info
    intraday_output_file_name = "intraday_momentum_strategy" + "_" + str(lookback_period) + "_" + str(lag_period) + "_" + \
                                str(lookforward_period) + "_" + "intraday_detail" + "_" + today_info
    
    output_file_pathname = os.path.join(output_file_path, output_file_name)
    intraday_output_file_pathname = os.path.join(output_file_path, intraday_output_file_name)
    
    with open(output_file_pathname,"w") as target_file, open(intraday_output_file_pathname,"w") as intraday_target_file:   
        
        # write header info into output file
        header = ["date","time_interval","lookback_period","lag_period","lookforward_period","percentile_rate",
                  "fee_rate","start_epochtime","end_epochtime","winner_cumulative_return","loser_cumulative_return"]
        intraday_header = ["date","time_interval","lookback_period","lag_period","lookforward_period","percentile_rate",
                           "fee_rate","time_stamp","winner_portfolio_initial_pos", "winner_portfolio_end_pos",
                           "winner_fee","loser_portfolio_initial_pos","loser_portfolio_end_pos","loser_fee"]
        
        target_file.write(",".join(map(str,header))+"\n")
        intraday_target_file.write(",".join(map(str,intraday_header))+"\n")
        
        # Iterate over all files 
        for filepath in file_path_list:
            start = time.time()
            try: 
                # Step 1: Read the target file as a dataframe
                data_df = read_stock_time_bar_to_csv(filepath, num_of_cols=10)
            
                # Step 2: Calculate return, append as a new column
                data_df = calculate_stock_return(data_df)
            
                # Step 3: retrieve time information for target_file
                date_info = dt.datetime.fromtimestamp(int(data_df.iloc[lookback_period+lag_period]["bar_Timestamp"]*\
                            (10**(-9)))).date()
                start_epoch_time = int(dt.datetime(date_info.year, date_info.month, date_info.day,
                                        start_time.hour, start_time.minute, start_time.second).timestamp())
                end_epoch_time = int(dt.datetime(date_info.year, date_info.month, date_info.day,
                                      end_time.hour, end_time.minute, end_time.second).timestamp())
            
                # Step 4: Apply Momentum Strategy simulator(will execute momentum strategy multiple times )
                momentum_portolio_return_dict, cumulative_daily_return_dict = momentum_strategy_simulator(data_df = data_df,
                                                                            start_time = start_epoch_time,
                                                                            end_time = end_epoch_time,
                                                                            time_interval = time_interval,
                                                                            lookback_period = lookback_period,
                                                                            lookforward_period = lookforward_period,
                                                                            lag_period = lag_period,
                                                                            percentile_rate = percentile_rate,
                                                                            fee_rate = fee_rate)

                winner_cumulative_return = cumulative_daily_return_dict.get("daily_winner_acc_return")
                loser_cumulative_return = cumulative_daily_return_dict.get("daily_loser_acc_return")
            
                # Step 5: Gather all information into a list
                return_info_list = [date_info, time_interval, lookback_period, lag_period, lookforward_period, percentile_rate,
                                  fee_rate, start_epoch_time, end_epoch_time, winner_cumulative_return, loser_cumulative_return]
            
                # Step 6: Write to target file
                target_file.write(",".join(map(str,return_info_list))+"\n")
                
                # Step 7: Write intraday level data into second output file
                for key,value in momentum_portolio_return_dict.items():
                    return_list = [date_info, time_interval, lookback_period, lag_period, lookforward_period,
                                   percentile_rate, fee_rate] + [int(key)] + list(value)
                    intraday_target_file.write(",".join(map(str,return_list))+"\n")
                
                end = time.time()
                print ("process finished for %s time elapsed:"% filepath, end - start)
                 
            except Exception as e:
                logger.error("operation failed for %s: " % filepath + str(e))
                end = time.time()
                print ("time elapsed:",end - start)
                continue
                
    if return_output== True:
        output_df = pd.read_csv(output_file_pathname, index_col=None, header=0) 
        return output_df


    
    
    
    
"""
Function 7: 
Statistical significance testing tool 1: non_overlap_momentum_strategy_ttest
input a momentum strategy simulation result (with detailed portfolios' value, transaction cost and timestamp)
compute the return rate of non-overlapping winner & loser & aggregate portfolio sets (i.e. each portfolio set
consist of return data of portfolios whose holding period does not overlap with each other)

"""


    
    
def non_overlap_momentum_strategy_ttest(target_dir, ttest_tail="two-sided", mode="normal"):

    """
        Arguments:
            target_dir -- the directory of the target file. The target file need to be a "intraday_detail" data file 
                          consisted of momentum strategy portfolio value's information on legit timepoints (i.e.timepoint
                          at which market data satisfy lookback & lookforward condition to construct momentum portfolios) 
            ttest_tail -- the tail condition used for a one-sample t test, value can be: [two-sided, less, greater]
            mode -- the strategy mode. return calculation in different modes are different, value can be:
                    I. normal -- regular momentum strategy where we buy winner portfolios and sell loser portfolios
                    II. contrarian -- contrarian strategy of momentum (which becomes "reversal" strategy) where we sell
                                      winner portfolios and buy loser portfolios
            
        Returns:
            processed_df -- dataframe with calculated return information
            t_test_result_dict -- dictionary of t-test results 
    
    """    
    # file directory check and input check
    if not os.path.exists(target_dir):
        raise ValueError("target file directory does not exist, please check")
    
    if mode not in ["normal","contrarian"]:
        raise ValueError("mode name is incorrect, can only be 'normal' or 'contrarian' ")
    
    if ttest_tail not in ["two-sided","less","greater"]:
        raise ValueError("ttest_tail name is incorrect, can only be 'two-sided', 'less' or 'greater' ")
    
    
    data_df = pd.read_csv(target_dir, index_col = None, header = 0)
    # initialized output dataframe's header info
    processed_df = pd.DataFrame(columns = data_df.columns)

    # iterate on each day of data (since there are large time gap between the end of a day and the start of the next day)
    # for each day, select data entry with non-overlapping holding period (i.e. use lookforward period as time interval)
    interval = data_df["lookforward_period"][0]
    date_info = list(data_df["date"].unique())
    for date in date_info:
        slice_df = data_df.loc[data_df["date"] == date]
        index_max = len(slice_df)
        for i in range(0,index_max,interval): 
            #print(slice_df.iloc[i])
            processed_df = processed_df.append(slice_df.iloc[i], ignore_index = True)

    processed_df.reset_index(drop=True)
    
    winner_init_pos = processed_df["winner_portfolio_initial_pos"]
    winner_end_pos = processed_df["winner_portfolio_end_pos"]
    winner_fee = processed_df["winner_fee"]
    loser_init_pos = processed_df["loser_portfolio_initial_pos"]
    loser_end_pos = processed_df["loser_portfolio_end_pos"]
    loser_fee = processed_df["loser_fee"]

    # normal mode: buy winner / sell loser at initial timepoint
    if mode == "normal":
        processed_df["winner_return_margin"] = winner_end_pos - winner_init_pos
        processed_df["loser_return_margin"] = loser_init_pos - loser_end_pos
        processed_df["winner_return_rate"] = (winner_end_pos - winner_init_pos - winner_fee)/ winner_init_pos
        processed_df["loser_return_rate"] = (loser_init_pos - loser_end_pos - loser_fee)/loser_init_pos
        processed_df["aggregate_return_rate"] = ((winner_end_pos - winner_init_pos - winner_fee) + \
                                             (loser_init_pos - loser_end_pos - loser_fee))/ (winner_init_pos + loser_init_pos)
    # contrarian mode: sell winner / buy loser at initial timepoint
    else:
        processed_df["winner_return_margin"] = winner_init_pos - winner_end_pos
        processed_df["loser_return_margin"] = loser_end_pos - loser_init_pos
        processed_df["winner_return_rate"] = (winner_init_pos - winner_end_pos - winner_fee)/ winner_init_pos
        processed_df["loser_return_rate"] = (loser_end_pos - loser_init_pos - loser_fee)/loser_init_pos
        processed_df["aggregate_return_rate"] = ((winner_init_pos - winner_end_pos - winner_fee) + \
                                             (loser_end_pos - loser_init_pos - loser_fee))/ (winner_init_pos + loser_init_pos)
    
    # perform one-sample t-test on winner & loser & aggregate portfolios 
    winner_ttest = stats.ttest_1samp(a = processed_df["winner_return_rate"],
                                     popmean = 0,
                                     alternative = ttest_tail)
    loser_ttest = stats.ttest_1samp(a = processed_df["loser_return_rate"],
                                    popmean = 0,
                                    alternative = ttest_tail)
    aggregate_strategy_ttest = stats.ttest_1samp(a = processed_df["aggregate_return_rate"],
                                                 popmean = 0,
                                                 alternative = ttest_tail)
    
    t_test_result_dict = {"winner_return_rate_ttest":winner_ttest,
                          "loser_return_rate_ttest":loser_ttest,
                          "aggregate_return_rate_ttest":aggregate_strategy_ttest}
    
    return processed_df, t_test_result_dict





    
"""
Function 8: 
Statistical significance testing tool 2: momentum_strategy_return_data_ttest
for all data files: input a momentum strategy simulation result (with detailed portfolios' value, transaction cost and timestamp)
compute the return rate of all winner & loser & aggregate portfolio sets (i.e. this test loosen the non-overlapping
requisite in non_overlap_momentum_strategy_ttest. Thus, this test reflects the general statistical significance of a 
particular momentum strategy's returns)

"""




def momentum_strategy_return_data_ttest(target_dir):
    """
        Arguments:
            target_dir -- the directory of the target file or folder consist of target files. The target file need to be a 
                          "intraday_detail" data file consisted of momentum strategy portfolio value's information on legit 
                          timepoints (i.e.timepoint at which market data satisfy lookback & lookforward condition to construct 
                          momentum portfolios) 
            
        Returns:
            return_df -- dataframe consist of one-sample t-test of one or multiple momentum strategies of non-overlapped holding
    
    """    
    
    file_path_list = []
    file_name_list = []
    
    # check target directory
    if os.path.isfile(target_dir) == True:
        if not target_dir.startswith('.'):
            file_path_list.append(target_dir)
            file_name_list.append(target_dir)
    elif os.path.isdir(target_dir) == True:
        for file in os.listdir(target_dir):
            if not file.startswith('.'):
                file_path_list.append(os.path.join(target_dir,file))
                file_name_list.append(file)
    else:
        raise ValueError("input is neither a filepath nor a directory, please check.")
    
    """
    # check output directory
    if not os.path.exists(output_file_path):
        raise ValueError("output_file_path does not exist, please check.") 
    """
    
    header = ["lookback_period","lag_period","lookforward_period","mode","ttest_tail",
              "winner_ttest_pvalue","loser_ttest_pvalue","aggregate_ttest_pvalue"]
    return_df = pd.DataFrame(columns=header)
    strategy_mode = ["normal","contrarian"]
    ttest_tail = ["less","greater"]
    
    # perform non_overlap_momentum_strategy_ttest on each file with different tails & strategy mode
    for i in range(len(file_path_list)):
        filepath = file_path_list[i]
        filename = file_name_list[i]
        #print(filename)
        
        with open(filepath,"r"):
            df = pd.read_csv(filepath, index_col = None, header = 0)
            lookback = df["lookback_period"][1]
            lag = df["lag_period"][1]
            lookforward = df["lookforward_period"][1]
            
        for mode in strategy_mode:
            for tail in ttest_tail:
                _, t_test_result_dict = non_overlap_momentum_strategy_ttest(filepath,
                                                                            ttest_tail= tail,
                                                                            mode= mode)
                ttest = t_test_result_dict.values()

                return_entry = [lookback, lag, lookforward, mode, tail,"%.6f"%list(ttest)[0][1],
                                float("%.6f"%list(ttest)[1][1]), float("%.6f"%list(ttest)[2][1])]
                    
                return_df.loc[len(return_df)] = return_entry
                    
    return_df = return_df.sort_values(by=["lookback_period","lag_period","lookforward_period","mode"])
                    
    return return_df 
    



    

"""
Function 9: 
return data regression analytical tool 1: momentum_return_regression
apply regression model of users' choice (OLS or AR2) over a particular kind of return (Winner portfolios/ Loser
Portfolios/ Aggregate Portfolios) and a particular kind of trading strategy (normal/ contrarian). Return: 
1. the original dataframe with calculated return information
2. the filtered dataframe with data entries that exceed the numerical threhold derived from a fitted regression model
3. the R-squared (coefficient of determination) score of the fitted regression model

"""    
    
    
    
    
def momentum_return_regression(target_dir, mode = "normal",regression_model = "OLS",
                               entry = "aggregate_return_rate", show_plot = False):

    """
        Arguments:
            target_dir -- the directory of the target file. The target file need to be a "intraday_detail" data file 
                          consisted of momentum strategy portfolio value's information on legit timepoints (i.e.timepoint
                          at which market data satisfy lookback & lookforward condition to construct momentum portfolios) 
            mode -- the strategy mode. return calculation in different modes are different, value can be:
                    I. normal -- regular momentum strategy where we buy winner portfolios and sell loser portfolios
                    II. contrarian -- contrarian strategy of momentum (which becomes "reversal" strategy) where we sell
                                      winner portfolios and buy loser portfolios
            regression_model -- choice of regression model. Value should be in ["OLS","AR2"], where:
                    I. OLS -- Ordinary Least Square model with one independent parameter and one constant parameter
                    II. AR2 -- AR(2) autoregression model
            entry -- choice of the data entry used for regression. Value should be in ["winner_return_rate",
                     "loser_return_rate","aggregate_return_rate"] where: 
                    I. winner_return_rate -- (run regression on) momentum winner return data
                    II. loser_return_rate -- (run regression on) momentum loser return data
                    III. aggregate_return_rate -- (run regression on) aggregate momentum return data (i.e. trade
                                                   both momentum winner and loser portfolio)
            show_plot -- Whether to show the graph that contains best fitting line, default is False
                                                  
        Returns:
            data_df -- dataframe derived from the data file in target_dir, with extra columns of return data 
            filtered_df -- consist of data filtered with threshold derived from a particular OLS or AR(2)
                           regression model, used for significance testing on profitability of a trading strategy
            adjusted_r_squared -- the adjusted r-squared score for the regression model
    
    """    
    # file directory check
    if not os.path.exists(target_dir):
        raise ValueError("target file directory does not exist, please check")
    
    if mode not in ["normal","contrarian"]:
        raise ValueError("mode name is incorrect, can only be 'normal' or 'contrarian' ")    
    
    data_df = pd.read_csv(target_dir, index_col = None, header = 0)
    
    # Step 1: calculate return margin and rates
    winner_init_pos = data_df["winner_portfolio_initial_pos"]
    winner_end_pos = data_df["winner_portfolio_end_pos"]
    winner_fee = data_df["winner_fee"]
    loser_init_pos = data_df["loser_portfolio_initial_pos"]
    loser_end_pos = data_df["loser_portfolio_end_pos"]
    loser_fee = data_df["loser_fee"]

    # normal mode:
    if mode == "normal":
        data_df["winner_return_margin"] = winner_end_pos - winner_init_pos
        data_df["loser_return_margin"] = loser_init_pos - loser_end_pos
        data_df["winner_return_rate"] = (winner_end_pos - winner_init_pos - winner_fee)/ winner_init_pos
        data_df["loser_return_rate"] = (loser_init_pos - loser_end_pos - loser_fee)/loser_init_pos
        data_df["aggregate_return_rate"] = ((winner_end_pos - winner_init_pos - winner_fee) + \
                                             (loser_init_pos - loser_end_pos - loser_fee))/ (winner_init_pos + loser_init_pos)
    # contrarian mode:
    else:
        data_df["winner_return_margin"] = winner_init_pos - winner_end_pos
        data_df["loser_return_margin"] = loser_end_pos - loser_init_pos
        data_df["winner_return_rate"] = (winner_init_pos - winner_end_pos - winner_fee)/ winner_init_pos
        data_df["loser_return_rate"] = (loser_end_pos - loser_init_pos - loser_fee)/loser_init_pos
        data_df["aggregate_return_rate"] = ((winner_init_pos - winner_end_pos - winner_fee) + \
                                             (loser_end_pos - loser_init_pos - loser_fee))/ (winner_init_pos + loser_init_pos)

    # Step 2: Run regression, calculate threshold for trading strategy
    # AR(1) Model with OLS optimization 
    if regression_model == "OLS":
        reg_x = np.array(data_df.iloc[0:-1][entry])
        reg_x = sm.add_constant(reg_x)
        reg_y = np.array(data_df.iloc[1:][entry])

        results = sm.OLS(endog=reg_y, exog=reg_x).fit()
        
        #print (results.summary())
        const = results.params[0]
        alpha = results.params[1]
        adjusted_r_squared = results.rsquared_adj
        
        if const == 0:
            threshold = 0
        else:
            threshold = (-const)/alpha
        
        if show_plot:
            x = np.array(data_df.iloc[0:-1][entry])
            plt.plot(x, reg_y, 'o')
            plt.plot(x, alpha*x + const)
            plt.title("OLS Regression Best Fitting line")
            plt.xlabel("X(t)")
            plt.ylabel("X(t+1)")
            
        # apply threshold on data_df, acquire filtered_df    
        filtered_df = pd.DataFrame(columns = data_df.columns)
        for i in range(len(data_df)-1):
            if data_df["aggregate_return_rate"][i] >= threshold:
                filtered_df = filtered_df.append(data_df.iloc[i+1])
    
    # AR(2) Model with MLE optimization         
    elif regression_model == "AR2":
        model = AutoReg(data_df[entry], 2, old_names=False)
        results = model.fit()
        #print (results.summary())
        const = results.params[0]
        alpha1 = results.params[1]   
        alpha2 = results.params[2]
        
        # Calculate R-squared manually
        y_true = data_df[entry][2:] 
        y_hat = np.array(alpha2 * data_df[entry][:-2]) + np.array(alpha1 * data_df[entry][1:-1]) + const
        sample_num = len(data_df[entry])
        y_mean = np.sum(data_df[entry][2:])/len(data_df[entry][2:])
        ss_res = np.sum(np.square(np.subtract(y_true,y_hat)))
        ss_tot = np.sum(np.square(np.subtract(y_true,y_mean)))
        r_squared = 1 - ss_res/ss_tot
        adjusted_r_squared = 1 - (1-r_squared)*((sample_num - 1)/(sample_num - len(results.params)-1))
        
        if show_plot:
            plot_acf(data_df[entry], lags=30)
            
        # apply threshold on data_df, acquire filtered_df  
        filtered_df = pd.DataFrame(columns = data_df.columns)
        for i in range(1,len(data_df)-1):
            if data_df["aggregate_return_rate"][i-1]*alpha2 + \
                data_df["aggregate_return_rate"][i]*alpha1 >= -const:
                filtered_df = filtered_df.append(data_df.iloc[i+1])
        
    # reindex and drop duplicated rows
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df = filtered_df.drop_duplicates()
    
    # Step 3: Perform right-tailed one-sample t-test on filtered_df 
    ttest = stats.ttest_1samp(filtered_df[entry],
                              popmean = 0,
                              alternative="greater")
    
    
    return data_df, filtered_df, adjusted_r_squared, ttest
        

    

    

"""
Function 10: 
return data regression analytical tool 2: momentum_strategy_simulator_with_regression
for all data files: 
1. fit a regression model, calculate numerical threshold from the regression parameters
2. apply numerical threshold on return data, acquire data entries from the simulation
3. perform right-tailed one-sample t-test on the return data acquire in step 2

"""
    
    
    

def momentum_strategy_simulator_with_regression(target_dir):
    """
        Arguments:
            target_dir -- the directory of the target file or folder consist of target files. The target file need to be a 
                          "intraday_detail" data file consisted of momentum strategy portfolio value's information on legit 
                          timepoints (i.e.timepoint at which market data satisfy lookback & lookforward condition to construct 
                          momentum portfolios) 
            
        Returns:
            return_df -- dataframe consist of the following information for one or multiple momentum strategies:
                         1. strategy information: lookback & lag & lookforward periods
                         2. strategy mode
                         3. regression information: model & return entry
                         4. accumulated return rate (use summation) resulted from the momentum strategy with signal indicator 
                            derived from regression model
                         5. r-squared score of the model
                         6. t-test statistics and p-value of the return data
    
    """  

    file_path_list = []
    file_name_list = []
    
    # check target directory
    if os.path.isfile(target_dir) == True:
        if not target_dir.startswith('.'):
            file_path_list.append(target_dir)
            file_name_list.append(target_dir)
    elif os.path.isdir(target_dir) == True:
        for file in os.listdir(target_dir):
            if not file.startswith('.'):
                file_path_list.append(os.path.join(target_dir,file))
                file_name_list.append(file)
    else:
        raise ValueError("input is neither a filepath nor a directory, please check.")
    
    """
    # check output directory
    if not os.path.exists(output_file_path):
        raise ValueError("output_file_path does not exist, please check.") 
    """
    header = ["lookback_period","lag_period","lookforward_period","mode","regression_model","entry",
              "adjusted_rsquared","accumulated_return_rate", "ttest_statistic","ttest_pvalue"]
    return_df = pd.DataFrame(columns=header)
    strategy_mode = ["normal","contrarian"]
    regression_model = ["OLS","AR2"]
    target_entry = ["winner_return_rate","loser_return_rate","aggregate_return_rate"]
    # for each datafile and each combinations of strategy & model & return, perform regression and significance testing using 
    for i in range(len(file_path_list)):
        filepath = file_path_list[i]
        filename = file_name_list[i]
        print(filename)
        
        with open(filepath,"r"):
            df = pd.read_csv(filepath, index_col = None, header = 0)
            lookback = df["lookback_period"][1]
            lag = df["lag_period"][1]
            lookforward = df["lookforward_period"][1]
        
        for mode in strategy_mode:
            for model in regression_model:
                for entry in target_entry:
                    _,filtered_df, adjusted_r_squared, ttest = momentum_return_regression(filepath, 
                                                                                          mode = mode,
                                                                                          regression_model = model,
                                                                                          entry = entry,
                                                                                          show_plot = False)
                    accumulated_return = np.sum(filtered_df[entry])
                    return_entry = [lookback, lag, lookforward, mode, model, entry, adjusted_r_squared,
                                    accumulated_return, float("%.6f"%list(ttest)[0]), float("%.6f"%list(ttest)[1])]
                    
                    return_df.loc[len(return_df)] = return_entry
                    
    return_df = return_df.sort_values(by=["lookback_period","lag_period","lookforward_period","mode","regression_model"])
                    
    return return_df    


