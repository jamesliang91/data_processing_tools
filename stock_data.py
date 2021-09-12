import os
import time
import numpy as np
import datetime as dt

class stock_tick (object):
    
    def __init__(self, EpochTime, MDStreamID, TimeStamp, ExchangeID, InstrumentID, PreClosePrice, OpenPrice,
                 ClosePrice, HighPrice, LowPrice, LastPrice, TradeCount, TotalTradeVolume, TotalTradeValue,
                 OpenInterest, TotalBidVolume, WeightedAvgBidPrice, TotalOfferVolume, WeightedAvgOfferPrice,
                 TopBidPrice, TopAskPrice):
        """
        Attributes:
            EpochTime -- The EpochTime Stamp of the beginning time point of a stock time bar
            MDStreamID -- Market status lable, e.g.: "C"--> close, "T" --> in active trading 
            TimeStamp -- human readable TimeStamp formatted on: year-month-day-hour-minute-second
            ExchangeID -- The ExchangeID of the asset
            InstrumentID -- The InstrumentID of the asset, known as "ticker"
            PreClosePrice -- The Close price of an asset from the previous day
            OpenPrice -- The Open Price of the asset on the corresponding day  
            ClosePrice -- The Close Price of a tick (can be zero even where there are transaction happened)
            HighPrice -- The High Price of the asset until the timepoint of tick 
            LowPrice -- The Low Price of the asset until the timepoint of tick 
            LastPrice -- The last trading price of the tick
            TradeCount -- cumulative number of transaction until the beginning timepoint of tick
            TotalTradeVolume -- The cumulative trading volume until the beginning timepoint of the tick
            TotalTradeValue -- The cumulative trading value (volume * weighted price) until the beginning timepoint of the tick
            OpenInterest -- the interest rate benchmark at the tick time
            TotalBidVolume -- The cumulative bid(buy) volume until the beginning timepoint of the tick
            WeightedAvgBidPrice -- The weighted averaged bid price at the tick time
            TotalOfferVolume -- The cumulative ask(sell) volume until the beginning timepoint of the tick
            WeightedAvgOfferPrice -- The weighted averaged offer price at the tick time 
            TopBidPrice -- the highest bid price at given stock tick
            TopAskPrice -- the lowest ask price at given stock tick
            
            *BidTenLevelVolume --
            *BidTenLevelValue --
            *OfferTenLevelVolume --
            *OfferTenLevelValue -- 
            
        """   
        
    
        self.EpochTime = EpochTime
        self.MDStreamID = MDStreamID
        self.TimeStamp = TimeStamp
        self.ExchangeID = ExchangeID
        self.InstrumentID = InstrumentID
        
        self.PreClosePrice = PreClosePrice
        self.OpenPrice = OpenPrice
        self.ClosePrice = ClosePrice
        self.HighPrice = HighPrice
        self.LowPrice = LowPrice  
        self.LastPrice = LastPrice
        
        self.TradeCount = TradeCount
        self.TotalTradeVolume = TotalTradeVolume     
        self.TotalTradeValue = TotalTradeValue

        self.OpenInterest = OpenInterest
        self.TotalBidVolume = TotalBidVolume    
        self.WeightedAvgBidPrice = WeightedAvgBidPrice
        self.TotalOfferVolume = TotalOfferVolume
        self.WeightedAvgOfferPrice = WeightedAvgOfferPrice        
        
        self.TopBidPrice = TopBidPrice
        self.TopAskPrice = TopAskPrice
    
    
    def get_info(self, attr = None):
        
        """
        Arguments:
            self -- a bar object
            tick_list -- attribute that want to look up, default value is None, if None returns all attributes
            
        Returns:
            output_bar: either the original bar
    
        """        
        
        if attr == None:
            return ({
                "tick_EpochTime": self.EpochTime,
                "tick_MDStreamID": self.MDStreamID,
                "tick_TimeStamp": self.TimeStamp,
                "tick_ExchangeID": self.ExchangeID,
                "tick_InstrumentID": self.InstrumentID,
        
                "tick_PreClosePrice": self.PreClosePrice,
                "tick_OpenPrice": self.OpenPrice,
                "tick_ClosePrice": self.ClosePrice,
                "tick_HighPrice": self.HighPrice,
                "tick_LowPrice ": self.LowPrice,
                "tick_LastPrice": self.LastPrice,
        
                "tick_TradeCount": self.TradeCount,
                "tick_TotalTradeVolume ": self.TotalTradeVolume,    
                "tick_TotalTradeValue": self.TotalTradeValue,
                
                "tick_OpenInterest": self.OpenInterest,
                "tick_TotalBidVolume": self.TotalBidVolume, 
                "tick_WeightedAvgBidPrice": self.WeightedAvgBidPrice,
                "tick_TotalOfferVolume": self.TotalOfferVolume,
                "tick_WeightedAvgOfferPrice": self.WeightedAvgOfferPrice,   
                
                "tick_TopBidPrice":self.TopBidPrice,
                "tick_TopAskPrice":self.TopAskPrice
            })
        
        else:
            attr = str(attr)
            assert attr in ["EpochTime", "MDStreamID", "TimeStamp", "ExchangeID", "InstrumentID", "PreClosePrice",
                            "OpenPrice", "ClosePrice", "HighPrice", "LowPrice", "LastPrice", "TradeCount", 
                            "TotalTradeVolume", "TotalTradeValue", "OpenInterest", "TotalBidVolume", "WeightedAvgBidPrice",
                            "TotalOfferVolume", "WeightedAvgOfferPrice","TopBidPrice","TopAskPrice"], "Incorrect Attribute Name!"
            
            return (getattr(self, attr))
    

    # Define a static methods of stock_tick class, which generate a stock_tick object 
    # instance from a list 
    def generate_stock_tick(data_list):
        
        assert (type(data_list)) == list, "Input is not a list"
        assert (len(data_list)) >= 19, "not enough elements, should have 19 elements"
            
        return (stock_tick(data_list[0],data_list[1],data_list[2],data_list[3],data_list[4],
                           data_list[5],data_list[6],data_list[7],data_list[8],data_list[9],
                           data_list[10],data_list[11],data_list[12],data_list[13],data_list[14],
                           data_list[15],data_list[16],data_list[17],data_list[18],data_list[23],data_list[43]))


class stock_time_bar (object):
    
    def __init__(self, Time, ExchangeID, InstrumentID, Open, Close, High, Low, BidAskMean, Volume, Bar_Volume=0):
        
        """
        Attributes:
            Time -- The EpochTime Stamp of the beginning time point of a stock time bar
            ExchangeID -- The ExchangeID of the asset
            InstrumentID -- The InstrumentID of the asset, known as "ticker"
            Open -- The Open Price of a stock time bar, notice that this value needs to be derived from LastPrice
                 of stock_tick instead of OpenPrice 
            Close -- The Close Price of a stock time bar, notice that this value needs to be derived from LastPrice
                  of stock_tick instead of ClosePrice
            High -- The High Price of a stock time bar (i.e. within the time interval defined by the bar), notice 
                 that this value needs to be derived from LastPrice of stock_tick instead of HighPrice
            Low --  The Low Price of a stock time bar (i.e. within the time interval defined by the bar), notice 
                 that this value needs to be derived from LastPrice of stock_tick instead of LowPrice
            BidAskMean -- The mean price of top bid & ask at the end of the stock bar (i.e.last tick in the stock bar)
            Volume -- The cumulative trading volume until the beginning timepoint of the stock_time_bar 
            Bar_Volume -- The trading volume within the time interval of the stock_time_bar
            
        """             
        
        self.Time = Time
        self.ExchangeID = ExchangeID
        self.InstrumentID = InstrumentID
        self.Open = Open
        self.Close = Close
        self.High = High
        self.Low = Low
        self.BidAskMean = BidAskMean
        self.Volume = Volume
        self.Bar_Volume = Bar_Volume
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self.idx += 1
        try:
            return self.data[self.idx-1]
        except IndexError:
            self.idx = 0
            raise StopIteration  # Done iterating.
    next = __next__
    
    

    def get_info(self, attr = None):
        
        """
        Arguments:
            self -- a bar object
            tick_list -- attribute that want to look up, default value is None, if None returns all attributes
            
        Returns:
            output_bar: either the original bar
    
        """        
        
        if attr == None:
            return ({
                "bar_Timestamp": self.Time,
                "bar_ExchangeID": self.ExchangeID,
                "bar_InstrumentID": self.InstrumentID,
                "bar_Open_Price": self.Open,
                "bar_Close_Price": self.Close,
                "bar_High_Price": self.High,
                "bar_Low_Price": self.Low,
                "bar_Top_Bid_Ask_Mean":self.BidAskMean,
                "bar_Cumulative_Volume": self.Volume,
                "bar_Volume": self.Bar_Volume
                
            })
        else:
            attr = str(attr)
            assert attr in ["Time", "ExchangeID", "InstrumentID", "Open", "Close",
                            "High", "Low", "BidAskMean", "Volume", "Bar_Volume"], "Incorrect Attribute Name!"
            return (getattr(self, attr))


    # Define a static methods of stock_time_bar class, which generate a stock_time_bar
    # object instance from a stock_tick       
        
    def initialize_stock_time_bar_from_tick(stock_tick):
        
        
        """
        Arguments:
            stock_tick -- a python list with tick data of one asset on one timespot, generated from 
                         original data file by readline() and split methods
            
        Returns:
            stock_time_bar: the stock time bar object after initialization
            
            
        stock bar attribute initialization logic:
            Time & ExchangeID & InstrumentID: same as tick
            Open = the Close of the last bar, here initialized as zero for further append
            Close = Close price of tick
            High = initialized as the close price of tick
            Low = initialized as the close price of tick
            BidAskMean = mean price of top bidprice and top askprice of tick
            Volume = Volume of tick
            Bar_Volume = 0 
    
        """        
        
        # if one of the bid_ask volumn is missing, use lastprice as bid ask mean
        bid_ask_lower_limit = 0.9 * float(stock_tick.LowPrice)
        if float(stock_tick.TopBidPrice) >= bid_ask_lower_limit and float(stock_tick.TopAskPrice) >= bid_ask_lower_limit:
            bid_ask_total = float(stock_tick.TopBidPrice) + float(stock_tick.TopAskPrice)
            BidAskMean_value = round(bid_ask_total/2,3)
        else:
            BidAskMean_value = float(stock_tick.LastPrice)
        
        
        return (stock_time_bar(Time = stock_tick.EpochTime,
                               ExchangeID = stock_tick.ExchangeID,
                               InstrumentID = stock_tick.InstrumentID,
                               Open = float(stock_tick.LastPrice),  # initialized as the current last price, will be appended 
                               Close = float(stock_tick.LastPrice), # initialized as the current last price, will be appended 
                               
                               High = float(stock_tick.LastPrice),
                               Low = float(stock_tick.LastPrice),
                               BidAskMean = BidAskMean_value,
                               Volume = int(stock_tick.TotalTradeVolume),
                               Bar_Volume = 0))
        
    # Define a static methods of stock_time_bar class, which generate a stock_time_bar
    # object instance from a stock_tick           
        
    def update_stock_time_bar_from_tick(self, stock_tick, time_interval = 300):
        
        
        
        """
        Arguments:
            self -- a stock time bar object
            stock_tick -- a python list with tick data of one asset on one timespot, generated from 
                         original data file by readline() and split methods
            time_interval -- time interval used in bar generation (unit: seconds)
            
        Returns:
            self: the stock time bar object after update
            
            
        stock bar attribute update logic:
            Close = the Close price of the current tick
            High = the maximum of (bar's Close Price(before update), tick's LastPrice)
            Low = the minimum of (bar's Close Price(before update), tick's LastPrice)
            BidAskMean = the new mean price of tick's top bidprice and top askprice
            Volume = the TotalTradeVolume of current tick
            Bar_Volume += the TotalTradeVolume of current tick - the Volume of current bar(before update)
            
         """
        
        
        assert int(self.InstrumentID) == int(stock_tick.InstrumentID), "asset does not match!"
        
        # bar_attr = ["Time", "ExchangeID", "InstrumentID", "Open", "Close", "High", "Low", "BidAskMean", "Volume", "Bar_Volume"]

        # check if time difference falls in interval
        #tick_time = int(int(stock_tick.EpochTime)*10**(-9))
        #bar_time = int(int(self.Time)*10**(-9))
        
        old_Close = float(self.Close)
        old_High = float(self.High)
        old_Low = float(self.Low)
        old_Volume = self.Volume
        old_Bar_Volume = self.Bar_Volume
        old_BidAskMean = self.BidAskMean
        
        bid_ask_lower_limit = 0.9 * old_Low

        # Note: Use LastPrice (NOT ClosePrice) from tick to append stock time bar attributes
        #if 0 < tick_time - bar_time < 300:
        self.Close = float(stock_tick.LastPrice)
        self.High = max(old_High, float(stock_tick.LastPrice))
        self.Low = min(old_Low, float(stock_tick.LastPrice))
        
        if float(stock_tick.TopBidPrice) >= bid_ask_lower_limit and float(stock_tick.TopAskPrice) >= bid_ask_lower_limit:
            bid_ask_total = float(stock_tick.TopBidPrice) + float(stock_tick.TopAskPrice)
            self.BidAskMean = round(bid_ask_total/2, 3)
        else:
            self.BidAskMean = old_BidAskMean
        
        self.Volume = int(stock_tick.TotalTradeVolume)
        self.Bar_Volume += int(stock_tick.TotalTradeVolume) - int(old_Volume)
            
        return self    
    
    
def stock_bar_generation(file_path, 
                 output_file_name,
                 output_target_path = "D:\\sample_data\\program_test_output",
                 time_interval = 300):
    """
        Arguments:
            file_path -- path of data file that have been processed through abnormal detection and stock data filter process
            output_file_name -- name for the ouput file
            output_target_path -- the target path to store your output file
            time_interval -- time interval used in bar generation (unit: seconds)
            
        Returns:
            output file path, output bar data will be wrote to this output file
            
            
        stock bar data generation and update logic:
            1. create an empty dictionary to store all stock_time_bar information for all asset on the given day
            2. read the header (columns' name) and write them to output file
            3. read each line of tick data, check if the asset exist in the dictionary (i.e. has key "uniqueID" (ExchangeID + InstrumentID))
               a. if exist, retrieve the corresponding value (a python list of stock_time_bar object)
                   i. if the tick falls in the time interval of the last bar object, use <update_stock_time_bar_from_tick]> method
                   ii. if not, use <initialize_stock_time_bar_from_tick> method
               b. if not exist, create a new key-value pair using the uniqueID as key, then initialize a list with one stock_time_bar object
            4. iterate over values of each key (asset) in the dictionary, write the bar data into output file
    
     """
        
    if not os.path.exists(file_path):
        return ("file path does not exist")
    
    today_info = str(dt.date.today())
    output_file_name = output_file_name + "_" + today_info
    output_file_path = os.path.join(output_target_path, output_file_name)
    
    if not os.path.exists(output_target_path):
        os.makedirs(output_target_path)
        
    stock_bar_dict = {}
    
    with open(file_path,"r") as raw_file, open(output_file_path,"w") as target_file:   
  
        # header = header.split(",")
        # length = len(header)
        header = raw_file.readline()
        header = ["bar_Timestamp","bar_ExchangeID","bar_InstrumentID","bar_Open_Price",
                  "bar_Close_Price","bar_High_Price","bar_Low_Price","bar_BidAskMean","bar_Cumulative_Volume","bar_Volume"]
        target_file.write(",".join(map(str,header))+"\n")
        # raw_file.seek(0,0)
        
        # read each line of data, select columns listed above; filter out data lines during inactive market period 
        for lineno, line in enumerate(raw_file):
            line_list = line.split(",")
            
            # use generate_stock_tick methods to generate a tick from a list
            tick_info = stock_tick.generate_stock_tick(line_list)
            
            # Check if corresponding stock already has bar list in stock bar dictionary.
            unique_ID = line_list[3] + line_list[4]
            
            if unique_ID in stock_bar_dict.keys():
                stock_bar_list = stock_bar_dict[unique_ID]
                last_bar = stock_bar_list[-1]
                
                # comparison logic: append or create new bar from tick
                tick_time = int(int(tick_info.EpochTime)*10**(-9))
                bar_time = int(int(last_bar.Time)*10**(-9))
                
                # if tick timestamp falls in the interval, use tick info to update bar 
                if 0 < tick_time - bar_time < time_interval:
                    stock_bar_list[-1] = last_bar.update_stock_time_bar_from_tick(tick_info, time_interval)
                
                # if tick timestamp falls outside the interval, create a new bar using the tick info
                # notice that open price of bar is append as the close price of last bar after bar being initialized 
                elif time_interval <= tick_time - bar_time:
                    stock_bar_list.append(stock_time_bar.initialize_stock_time_bar_from_tick(tick_info))
                    stock_bar_list[-1].Open = stock_bar_list[-2].Close
            
            # if uniqueID (i.e.the asset) does not exist in stock_bar_dict, then create one new key-value pair
            else:
                stock_bar_dict[unique_ID] = [stock_time_bar.initialize_stock_time_bar_from_tick(tick_info)]
        
        for sub_dict in stock_bar_dict:
            for bar in stock_bar_dict.get(sub_dict):
                target_file.write( ",".join(map(str,list(bar.get_info().values()))) + "\n")
            # for debugging and quick check, uncomment the line below                  
            # print (str(sub_dict) + ": " + str([ bar.get_info("Close") for bar in stock_bar_dict.get(sub_dict)]))
    
    return ("process completed, please check output data file at: " + str(output_file_path))
    

    