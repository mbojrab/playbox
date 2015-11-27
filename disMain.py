'''The purpose of this program is to find some optimal parameter
   settings for a given program. If you expose parameters to the
   command line, but don't know what the optimal default value
   should be, this program will distribute the parameter
   possibilities over a cluster of machines, and rank the results
   based on user defined criteria.

   dispynode.py must be run on all machines in the cluster prior to
   running this program. The machines will be identified
   automatically, and made available for processing.

   The user must specify a program, which is assumed to be network
   accessible to all nodes of the cluster. The user must also specify
   a argument file. This file is assumed to be a text file with all
   of the parameter sets the user wants to test, one per line. The
   program will handle distribution over the cluster, and returning
   the results. 
   
   If the user needs additional functionality, they can derive the
   JobRunner class with some additional functionality specific to 
   their needs.
'''
class JobRunner() :
    
    def __init__(self, jobID, prog, argList) :
        '''Setup some common information for the class. This object is
           distributed with the job to the target node on the cluster. It
           is used as a conduit for passing information around the cluster
           and collecting results later.

           prog     - program to run (assumed to be network accessible)
           argList  - unique list of arguments to run on this node
           jobID    - unique identifier for this job (numeric)
           hostname - name of the machine this job was ran on
           stdout   - standard output generated by the subprocess
           stderr   - standard error generated by the subprocess
           results  - results should contain a metric for which the host 
                      can later rank the job results.
        '''
        self.prog = prog
        self.argList = argList if not isinstance(argList, str) \
                       else argList.split()
        self.jobID = jobID
        self.hostname = None
        self.stdout = None
        self.stderr = None
        self.result = None
        
    def __str__(self) :
        return 'Job [' + str(self.jobID) + ']\n' + \
               '\tProgram  = ' + str(self.prog) + '\n' + \
               '\tArgList  = ' + str(self.argList) + '\n' + \
               '\tHostName = ' + str(self.hostname) + '\n' + \
               '\tStdOut   = ' + str(self.stdout) + '\n' + \
               '\tStdErr   = ' + str(self.stderr) + '\n' + \
               '\tResult   = ' + str(self.result) + '\n'

    # run this command as a subprocess on the node
    def _runSubProcess(self) :
        from subprocess import Popen
        import socket
        self.hostname = socket.gethostname()
        (self.stdout, self.stderr) = Popen(args=[self.prog] + self.argList,
                                           stdout=Popen.PIPE, stderr=Popen.PIPE,
                                           shell=False).communicate()

    # collect the results for this run
    def _collectResults(self) :
        # by default we assume the program returns a numeric metric
        # for ranking to standard output
        self.result = float(self.stdout)
        
    def run(self) :
        self._runSubProcess()
        self._collectResults()

    @staticmethod
    def rank(jobs) :
        lowestJob = None
        for job in jobs :
            if lowestJob is None or job.result < lowestJob.result :
                lowestJob = job
        return lowestJob

def disCompute(jobRunner) :
    # run the provided class
    jobRunner.run()
    return "here"
    #return jobRunner.hostname


if __name__ == '__main__':
    import dispy, dispy.httpd, argparse, logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--prog', dest='prog',
                        help='Program to run in the subprocess.')
    parser.add_argument('--args', dest='args', 
                        help='File containing all program variations (one per line)')
    parser.add_argument('--override', dest='jobRunner', 
                        help='Python file containing a derived JobRunner')
    options = parser.parse_args()

    # setup a typedef for the new runner    
    JR = JobRunner
    if options.jobRunner is not None :
        cls = options.jobRunner
        imp = __import__(cls)
        JR = imp.cls

    # setup the logger
    log = logging.getLogger('dispy: ' + options.prog)
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    stream.setFormatter(formatter)
    log.addHandler(stream)

    # open the argument list for parsing
    with open(options.args, 'r') as f :
        argSets = f.readlines()

    # create the cluster for this program launch
    cluster = dispy.JobCluster(disCompute, depends=[JR])

    # start a monitor for the cluster at http://localhost:8181
    jobMonitor = dispy.httpd.DispyHTTPServer(cluster, host='localhost')

    # submit the jobs to the cluster --
    # this submits each argument set individually, and stores the jobs
    # so we can later recall their outputs and status. There is no
    # assignment here to schedule on the next available node.
    jobs = []
    for ii in range(len(argSets)) :
        log.info('Adding Job[' + str(ii) + ']')

        # create an object to perform the work on the node
        runner = JR(ii, options.prog, argSets[ii])
        job = cluster.submit(runner)
        job.id = runner
        jobs.append(job)

    # wait for the cluster to finish
    cluster.wait()

    # print the statistic of the clustered run
    log.info(cluster.stats())

    for job in jobs :
        log.info('Job[' + str(job.id.jobID) + ']: ' + str(job.result))
    
    # rank the result and return the best
    bestJob = JR.rank(jobs)
    log.info('The best job was: \n' + str(bestJob.result))

    # cleanup
    jobMonitor.shutdown()
    cluster.close()