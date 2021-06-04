import logging, os
import logging.config



def singleton(cls):
    instances = {}
    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance()

@singleton
class Logger():
    def __init__(self):
        print(os.getcwd())
        dirname, algoName = os.path.split(os.path.abspath(os.getcwd()))
        logFile = os.path.join(os.getcwd(), r'ML_Workbench_{}.log'.format(algoName))
        logging.basicConfig(filename=logFile,
                            format='%(asctime)s  %(levelname)s\t%(filename)s : %(funcName)s : %(lineno)04d \t %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filemode='w',
                            level=logging.DEBUG
                            )
        self.logger = logging.getLogger(__name__)
