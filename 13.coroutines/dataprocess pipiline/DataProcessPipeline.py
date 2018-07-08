# this file define a data process pipeline to process the data 
# use generator to do the watching function&let the data process through the pipline 
import time 
import re 
import threading 
import queue 
# generator function 
# filter function(provide the key,filter the value with given functions) 
def field_map(dict, name, func): 
    for d in dict: 
        d[name] = func(d[name]) 
        yield d 

# generator function 
# watch a log file (like tail -f) 
# get the last line of a file 
# yield any new line that wirte to the bottom of the file 
# this function serve as a data provider in a pipeline filter structure 
def watch(file): 
    # goto the end of a file 
    file.seek(0, 2) 
    while(True): 
        line = file.readline() 
        if not line: 
            # no new line now,just wait a while break and get another try(can be put into a coroutines task) 
            time.sleep(0.1) 
            print("watching...") 
            continue 

        yield line 

# parse the data line 
# this function serve as a data processer (a collection of filters) in a pipeline filter structure 
def parse(lines): 
    # filter1 use re to parse the log 
    logpats = r'(\S+) (\S+) (\S+) \[(.*?)\] '\ r'"(\S+) (\S+) (\S+)" (\S+) (\S+)' 
    logpat = re.compile(logpats) 
    groups = (logpat.match(line) for line in lines) 

    # filter2 convert from groups to tuples 4 further process 
    tuples = (g.groups() for g in groups if g) 
    
    # filter3 generate dictionaries(key = colname,value = tupleval) 
    colnames = ('host','referrer','user','datetime','method', 'request','proto','status','bytes')
    log = (dict(zip(colnames, tuple)) for tuple in tuples) 

    # filter4 use the field to filter the certain column value 
    # change the status value 2 int 
    log = field_map(log, "status", int) 
    
    # correct the error values 
    log = field_map(log, "bytes", lambda s: int(s) if s !='-' else 0) 
    
    # return the generator object(not yield! because the "parse" function is not a generator) 
    return log 

# broadcast (one dataprovider to several listeners) 
# go throuth the whole list to get the send message and the consumers thread(use thread to solve the parallel requirements) 
# coroutines is another option consumer1(function) to query and print the 404 
#def find404(log): 
#    for r in (r for r in log if r['s

