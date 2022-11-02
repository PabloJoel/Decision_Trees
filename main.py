import pandas as pd
from DecisionTree import DecisionTree

if __name__ == '__main__':
    data = pd.read_csv('regar.txt')

    dt = DecisionTree()
    tree = dt.execute(data, 'Regar')

    print(tree)
