from timeit import default_timer as timer
from six.moves.queue import LifoQueue as queue
import lxml.etree as et
import logging
import atexit

def setupLogging (options, appName) :
    '''Grab the logger and parser from the options.'''
    import logging
    from builder.profiler import Profiler

    logName = appName + ': ' + options.data

    # setup the logger
    log = logging.getLogger(logName)
    log.setLevel(options.level.upper())
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream = logging.StreamHandler()
    stream.setLevel(options.level.upper())
    stream.setFormatter(formatter)
    log.addHandler(stream)

    # attach it to a file -- if requested
    if options.logfile is not None :
        logFile = logging.FileHandler(options.logfile)
        logFile.setLevel(options.level.upper())
        logFile.setFormatter(formatter)
        log.addHandler(logFile)

    # setup the profiler
    prof = Profiler(log=log, name=logName, profFile=options.profile)

    return log, prof

class Profiler () :
    def __init__ (self, log=None, name='ApplicationName',
                  profFile='./ApplicationName-Profile.xml') :
        if log is not None and not isinstance(log, logging.Logger) :
            raise ValueError("'log' must be a valid logging.Logger type")
        self._profileStack = queue()
        self._profileFile = profFile
        self._root = et.Element('ApplicationProfile')
        self._activeElem = self._root
        self._log = log
        
        # start a profile for the program level
        self.startProfile(name, 'critial')
        atexit.register(self.cleanup)

    def startProfile (self, message, level='debug') :
        '''Add a profile to the stack. This is a nested call in lifo order'''
        self._logMessage(message, level)
        
        # add this to the stack for later recall
        self._profileStack.put_nowait({'message':message,
                                       'startTime':timer()})

        # make an element for this message and update the activeElem
        newElem = et.Element('profile')
        self._activeElem.append(newElem)
        self._activeElem = newElem

    def endProfile (self) :
        '''Pop the latest profile and log it and its execution time'''
        if self._profileStack.empty() :
            raise LookupError('There are no profiles in the queue.')

        # pop the profile off the stack and update its attributes
        recentProf = self._profileStack.get()
        self._activeElem.attrib['time'] = \
            str(timer() - recentProf['startTime']) + 's'
        self._activeElem.attrib['message'] = recentProf['message']

        # reset to activeElem to its parent
        self._activeElem = self._activeElem.getparent()

    def _logMessage(self, message, level) :
        '''Log the message at the specified level'''
        if self._log is not None :
            try :
                # try logging the message at the requested level
                getattr(self._log, level)(message)
            except :
                # just log it to debug
                self._log.debug(message)

    def cleanup(self) :
        '''Write the profile stack to a file'''
        if self._profileFile is not None :
            # close up any loose ends
            while not self._profileStack.empty() :
                self.endProfile()
            # write the xml to disk
            with open(self._profileFile, 'wb') as f :
                f.write(et.tostring(self._root, pretty_print=True))

if __name__ == '__main__' :
    import time
    log = logging.getLogger('profilerTest')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    stream.setFormatter(formatter)
    log.addHandler(stream)

    prof = Profiler(log, "test", 'E:/out.xml')
    prof.startProfile('testing', 'info')
    prof.startProfile('testingAgain', 'debug')
    time.sleep(1)
    prof.endProfile()
    prof.endProfile()
