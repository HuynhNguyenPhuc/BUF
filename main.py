from preprocessing import preprocessing
from api import ModelManager
from evaluation import evaluation

import sys
from utils.datetime import is_valid_date

def printUsage():
    print("""
    Usage: py main.py <command>
    Commands:
    + process: Perform preprocessing
    + train: Train the model
    + predict <date> <exchange> <stock>: Predict stock prices in the future
        <date>: Date in the format 'dd/mm/yyyy'
        <exchange>: Stock exchange name
        <stock>: Stock symbol
    + plot <date> <exchange> <stock> - Plot stock prices
        <date>: Date in the format 'dd/mm/yyyy'
        <exchange>: Stock exchange name
        <stock>: Stock symbol
    + evaluation: Perform evaluation
    """)

def main(argv):
    num_params = len(argv)

    if num_params == 1:
        printUsage()
    elif num_params == 2:
        if argv[1] == "process":
            preprocessing()
        elif argv[1] == "evaluation":
            evaluation()
        else:
            print("Invalid command:", argv[1])
            printUsage()
    elif num_params >= 3:
        manager = ModelManager()
        if argv[1] == "train":
            manager.train()
        elif argv[1] == "predict":
            if len(argv) < 5:
                print("Insufficient parameters for predict command")
                printUsage()
            else:
                date = argv[2]
                exchange = argv[3]
                stock = argv[4]
                if is_valid_date(date):
                    manager.stock_predict(stock, exchange, date)
                    printUsage()
        elif argv[1] == "plot":
            if len(argv) < 5:
                print("Insufficient parameters for plot command")
                printUsage()
            else:
                date = argv[2]
                exchange = argv[3]
                stock = argv[4]
                if is_valid_date(date):
                    manager.plot(stock, exchange, date)
                else:
                    print("Invalid parameters for plot command")
                    printUsage()
        else:
            print("Invalid command:", argv[1])
            printUsage()

if __name__ == "__main__":
    main(sys.argv)