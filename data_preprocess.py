import os
import time
import datetime as dt


def abnormal_data_filter(file_path,output_file_name, output_target_path = "D:\\sample_data\\program_test_output"):
    
    """
    Arguments:
        file_path -- target file that need to be processed
        output_file_name -- a file name for your output file
        output_target_path -- the target directory where you want to store your output file
    
    Return:
        None, this function is used as a data processor for the generation of bar data of stocks
    
    """
    
    if not os.path.exists(file_path):
        return ("file path does not exist")
    
    today_info = str(dt.date.today())
    output_file_name = output_file_name + "_" + today_info
    output_file_path = os.path.join(output_target_path, output_file_name)
    
    if not os.path.exists(output_target_path):
        os.makedirs(output_target_path)
        
    with open(file_path,"r") as raw_file, open(output_file_path,"w") as target_file:        
        # check data abnormality on each line
        
        header = raw_file.readline()
        target_file.write(header)
        length = len(header.split(","))
        raw_file.seek(0,0)
        
        for lineno, line in enumerate(raw_file):
            
            # 1. filter out incomplete lines (not enough entries as header) of data
            line_list = line.split(",")
            if len(line_list) < length:
                continue
                
            # 2. filter out lines with non-numerical value, negative values or empty values ("") 
            # in entries of price & volume (all columns except 1-5th columns)
            error_flag = False
            for entry in line_list[5:]:
                try:
                    float(entry)
                    if float(entry)< 0:
                        error_flag = True
                        break
                except ValueError:
                    error_flag = True
                    break
                    
            if error_flag:
                continue
            else:
                target_file.write(line)
            
    print ("number of columns: ", length)        
    print ("process finished, please check output file in the following directory: " + output_file_path)
    
    
    
# [stock_filter] method is used to filter out data of non-stock assets from the whole-market-data file

def stock_bar_data_filter(file_path, output_file_name, output_target_path = "D:\\sample_data\\program_test_output", bid_offer_spread = True):
    
    """
    Arguments:
        file_path -- target file that need to be processed
        output_file_name -- a file name for your output file
        bid_offer_spread = whether to preserve 10-level bid-ask spread information or not, True (preserve) or False (not preserve)
        output_target_path -- the target directory where you want to store your output file
    
    Return:
        None, this function is used as a data processor for the generation of bar data of stocks
    
    """
    
    if not os.path.exists(file_path):
        return ("file path does not exist")
    
    today_info = str(dt.date.today())
    output_file_name = output_file_name + "_" + today_info
    output_file_path = os.path.join(output_target_path, output_file_name)
    
    if not os.path.exists(output_target_path):
        os.makedirs(output_target_path)
    
    with open(file_path,"r") as raw_file, open(output_file_path,"w") as target_file:   
        # use the 4th attribute "ExchangeId" along with the 5th attribute "InstrumentID" and apply the coding rule of 
        # shanghai & shenzhen exchange over stock asset to filter out non-stock asset, coding rules can be referred above        
        header = raw_file.readline()
        header = header.split(",")
        length = len(header)
        if bid_offer_spread == True:
            target_file.write(",".join(map(str,header[0:8]+header[10:19]+header[20:22]+header[23:63]))+"\n")
        else:
            target_file.write(",".join(map(str,header[0:8]+header[10:19]+header[20:22]))+"\n")
        raw_file.seek(0,0)
        
        # read each line of data, select columns listed above; filter out data lines during inactive market period 
        for lineno, line in enumerate(raw_file):
            line_list = line.split(",")
            if len(line_list) < length:
                continue
            ExchangeID = str(line_list[3])
            InstrumentID = str(line_list[4])
            if bid_offer_spread == True:
                output_line = ",".join(map(str,line_list[0:8]+line_list[10:19]+line_list[20:22]+line_list[23:63]))
            else:
                output_line = ",".join(map(str,line_list[0:8]+line_list[10:19]+line_list[20:22]))
            
            # Check if the timestamp of data line is in the active market period
            # Note: use datetime.time() to compare only time information, not date information!
            
            if lineno <= 1:
                continue
            else:
                str_time = dt.datetime.fromtimestamp(int(int(line_list[0])*10**(-9)))
                if str_time.time() < dt.datetime(2021,6,1,9,30).time():
                    continue
                elif dt.datetime(2021,6,1,11,30).time() < str_time.time() < dt.datetime(2021,6,1,13,0).time():
                    continue
                elif dt.datetime(2021,6,1,14,57).time() < str_time.time():
                    continue
                
            
            # Shanghai Exchange, stock instrumentID should be 6-digits start with 6
            if ExchangeID == "SSE" and len(InstrumentID) == 6: 
                if InstrumentID[0] == "6":
                    target_file.write(output_line + "\n")
                else:
                    continue
                    
            # Shenzhen Exchange, stock instrumentID should be 6-digits start with 000 or 300
            elif ExchangeID == "SZE" and len(InstrumentID) == 6: 
                if InstrumentID[0:3] in ["000","300"]:
                    target_file.write(output_line + "\n")
                else:
                    continue
                    
            else:
                continue
                
    print ("number of columns: ", length)        
    print ("process finished, please check output file in the following directory: " + output_file_path)