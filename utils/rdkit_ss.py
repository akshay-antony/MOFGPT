import csv
import glob
import os
import sys
import yaml
import json
from rdkit import Chem

if __name__ == "__main__":
    x = Chem.MolFromSmiles("CCO1NCNCC1C2CNCNCCNCCNCCNC.CO1C2CC(1CC(2()C=)(CCNCCC(OC)CC(O)))C=ccc1C=")
    print(x)
    # for key in x.GetPropsAsDict():
    #     print(key)
    pass