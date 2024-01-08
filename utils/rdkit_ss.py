import csv
import glob
import os
import sys
import yaml
import json
from rdkit import Chem

if __name__ == "__main__":
    x = Chem.MolFromSmiles("CCC1CC2CCC(1C2C)1CNC[CH]CCCC(C(OCO)C)[CH]C(O)OCCCC(O)OCC")
    print(x)
    # for key in x.GetPropsAsDict():
    #     print(key)
    pass