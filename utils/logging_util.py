import os
import re
import time
import stat
import logging
import logging.handlers

_MIDNIGHT = 86400

class MultiThreadTimedRotatingFileHandler(logging.handlers.BaseRotatingHandler):
    """
    Handler for logging to a file, rotating the log file at certain timed
    intervals.

    If backupCount is > 0, when rollover is done, no more than backupCount
    files are kept - the oldest ones are deleted.
    """
    def __init__(self, filename, when='h', interval=1, backupCount=0, 
                 encoding=None, delay=False, utc=False, atTime=None):       
        self.when = when.upper()
        self.backupCount = backupCount
        self.utc = utc
        self.atTime = atTime
        # Calculate the real rollover interval, which is just the number of
        # seconds between rollovers.  Also set the filename suffix used when
        # a rollover occurs.  Current 'when' events supported:
        # S - Seconds
        # M - Minutes
        # H - Hours
        # D - Days
        # midnight - roll over at midnight
        # W{0-6} - roll over on a certain day; 0 - Monday
        #
        # Case of the 'when' specifier is not important; lower or upper case
        # will work.
        if self.when == 'S':
            self.interval = 1 # one second
            self.suffix = '%Y-%m-%d_%H-%M-%S.log'
            self.extMatch = r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.log$'
        elif self.when == 'M':
            self.interval = 60 # one minute
            self.suffix = "%Y-%m-%d_%H-%M.log"
            self.extMatch = r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$'
        elif self.when == 'H':
            self.interval = 60 * 60 # one hour
            self.suffix = '%Y-%m-%d_%H.log'
            self.extMatch = r'^\d{4}-\d{2}-\d{2}_\d{2}.log$'
        elif self.when == 'D' or self.when == 'MIDNIGHT':
            self.interval = 60 * 60 * 24 # one day
            self.suffix = '%Y-%m-%d.log'
            self.extMatch = r'^\d{4}-\d{2}-\d{2}.log$'
        elif self.when.startswith('W'):
            self.interval = 60 * 60 * 24 * 7 # one week
            if len(self.when) != 2:
                raise ValueError("You must specify a day for weekly rollover from 0 to 6 (0 is Monday): %s" % self.when)
            if self.when[1] < '0' or self.when[1] > '6':
                raise ValueError("Invalid day specified for weekly rollover: %s" % self.when)
            self.dayOfWeek = int(self.when[1])
            self.suffix = '%Y-%m-%d.log'
            self.extMatch = r'^\d{4}-\d{2}-\d{2}.log$'
        else:
            raise ValueError("Invalid rollover interval specified: %s" % self.when)

        #self.baseFilename = filename 
        #self.currentFileName = self._compute_fn() 
        logging.handlers.BaseRotatingHandler.__init__(self, filename, 'a', encoding, delay)
        self.extMatch = re.compile(self.extMatch, re.ASCII)
        self.interval = self.interval * interval # multiply by units requested
        if os.path.exists(filename):
            t = os.stat(filename)[stat.ST_MTIME]
        else:
            t = int(time.time())
        self.rolloverAt = self.computeRollover(t)

    def compute_fn(self): 
        return self.baseFilename + "." + time.strftime(self.suffix, time.localtime())

    def _open(self): 
        #if self.stream:
        #    self.stream.close()
        #    self.stream = None

        stream = open(self.compute_fn(), self.mode, encoding=self.encoding) 
        # simulate file name structure of `logging.TimedRotatingFileHandler` 
        #if os.path.exists(self.baseFilename): 
        #    try: 
        #        os.remove(self.baseFilename) 
        #    except OSError: 
        #        pass 
        #    try: 
        #        os.symlink(self.currentFileName, self.baseFilename) 
        #    except OSError: 
        #        pass 
        return stream

    def computeRollover(self, currentTime):
        """
        Work out the rollover time based on the specified time.
        """
        result = currentTime + self.interval
        # If we are rolling over at midnight or weekly, then the interval is already known.
        # What we need to figure out is WHEN the next interval is.  In other words,
        # if you are rolling over at midnight, then your base interval is 1 day,
        # but you want to start that one day clock at midnight, not now.  So, we
        # have to fudge the rolloverAt value in order to trigger the first rollover
        # at the right time.  After that, the regular interval will take care of
        # the rest.  Note that this code doesn't care about leap seconds. :)
        if self.when == 'MIDNIGHT' or self.when.startswith('W'):
            # This could be done with less code, but I wanted it to be clear
            if self.utc:
                t = time.gmtime(currentTime)
            else:
                t = time.localtime(currentTime)
            currentHour = t[3]
            currentMinute = t[4]
            currentSecond = t[5]
            currentDay = t[6]
            # r is the number of seconds left between now and the next rotation
            if self.atTime is None:
                rotate_ts = _MIDNIGHT
            else:
                rotate_ts = ((self.atTime.hour * 60 + self.atTime.minute)*60 +
                    self.atTime.second)

            r = rotate_ts - ((currentHour * 60 + currentMinute) * 60 +
                currentSecond)
            if r < 0:
                # Rotate time is before the current time (for example when
                # self.rotateAt is 13:45 and it now 14:15), rotation is
                # tomorrow.
                r += _MIDNIGHT
                currentDay = (currentDay + 1) % 7
            result = currentTime + r
            # If we are rolling over on a certain day, add in the number of days until
            # the next rollover, but offset by 1 since we just calculated the time
            # until the next day starts.  There are three cases:
            # Case 1) The day to rollover is today; in this case, do nothing
            # Case 2) The day to rollover is further in the interval (i.e., today is
            #         day 2 (Wednesday) and rollover is on day 6 (Sunday).  Days to
            #         next rollover is simply 6 - 2 - 1, or 3.
            # Case 3) The day to rollover is behind us in the interval (i.e., today
            #         is day 5 (Saturday) and rollover is on day 3 (Thursday).
            #         Days to rollover is 6 - 5 + 3, or 4.  In this case, it's the
            #         number of days left in the current week (1) plus the number
            #         of days in the next week until the rollover day (3).
            # The calculations described in 2) and 3) above need to have a day added.
            # This is because the above time calculation takes us to midnight on this
            # day, i.e. the start of the next day.
            if self.when.startswith('W'):
                day = currentDay # 0 is Monday
                if day != self.dayOfWeek:
                    if day < self.dayOfWeek:
                        daysToWait = self.dayOfWeek - day
                    else:
                        daysToWait = 6 - day + self.dayOfWeek + 1
                    newRolloverAt = result + (daysToWait * (60 * 60 * 24))
                    if not self.utc:
                        dstNow = t[-1]
                        dstAtRollover = time.localtime(newRolloverAt)[-1]
                        if dstNow != dstAtRollover:
                            if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                                addend = -3600
                            else:           # DST bows out before next rollover, so we need to add an hour
                                addend = 3600
                            newRolloverAt += addend
                    result = newRolloverAt
        return result

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.

        record is not used, as we are just comparing times, but it is needed so
        the method signatures are the same
        """
        t = int(time.time())
        if t >= self.rolloverAt:
            return 1
        return 0

    def getFilesToDelete(self):
        """
        Determine the files to delete when rolling over.

        More specific than the earlier method, which just used glob.glob().
        """
        dirName, baseName = os.path.split(self.baseFilename)
        fileNames = os.listdir(dirName)
        result = []
        prefix = baseName + "."
        plen = len(prefix)
        for fileName in fileNames:
            if fileName[:plen] == prefix:
                suffix = fileName[plen:]
                if self.extMatch.match(suffix):
                    result.append(os.path.join(dirName, fileName))
        result.sort()
        if len(result) < self.backupCount - 1:
            result = []
        else:
            result = result[:len(result) - self.backupCount + 1]
        return result

    def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        #dfn = self.rotation_filename(self.baseFilename + "." +
        #                             time.strftime(self.suffix, timeTuple))
        #if os.path.exists(dfn):
        #    os.remove(dfn)
        #self.baseFilename = self.compute_fn()
        #self.rotate(self.baseFilename, dfn)
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        #If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:           # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt

class LoggingUtil():
    def __init__(self, log_file_path=None, level='NOTSET'):
        logger = logging.getLogger()
        logger.setLevel(level=level)

        if log_file_path is None:
            LOG_PATH = os.path.join(
                os.path.abspath(os.path.dirname(__file__)), 
                'log/dnhpx_cv_log'
            )
        else:
            LOG_PATH = log_file_path
        #LOG_DIR = os.path.join(CURRENT_DIR, log_file_name)

        if not os.path.exists(os.path.dirname(LOG_PATH)):
            os.makedirs(os.path.dirname(LOG_PATH))
    
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(lineno)s - %(levelname)s - %(message)s'
        )
        handler = MultiThreadTimedRotatingFileHandler(
            filename=LOG_PATH, when='D', interval=1, backupCount=14, encoding='utf-8'
        )
        handler.setFormatter(formatter)
        handler.setLevel(logging.NOTSET)
    
        console = logging.StreamHandler()
        console.setLevel(logging.NOTSET)

        logger.addHandler(handler)
        logger.addHandler(console)

    def getLogger(self, name):
        logger = logging.getLogger(name)
        return logger