"""Log Directory"""

import shutil
import os
from datetime import datetime
from function_telegram_bot import telegram_bot_text


def log_directory(logdir=0, logname="coccobello"):
   "save the log of the training"
   path = "logs/log_example"
   if logdir == -1: #for deleting existing log files
      if os.path.exists(path) and os.path.isdir(path): #to verify if the folder exists and it is a directory or not
         shutil.rmtree(path) #permanently removes past files
   elif logdir == 0: #for using a temporary log file
      return path
   elif logdir == 1: #for using a scheduled log file
      return "logs/log_%s_" %(logname) + datetime.now().strftime("%Y%m%d_%H%M%S")

def log_zip(logpath, file_name="coccobello"):
   "create results zips"
   shutil.make_archive(logpath + "/%s_train" %(file_name), 'zip', logpath + "/train") #creates zip for train results
   try: shutil.make_archive(logpath + "/%s_validation" %(file_name), 'zip', logpath + "/validation") #creates zip for validation results
   except Exception: pass
