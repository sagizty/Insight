import numpy as np
import os

data_path = '/home/NSCLC-project/NSCLC_go/3D_local_dataset'

if __name__ == "__main__":
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            direction = os.path.append(data_path, file)
            f_data = np.load(direction)
            print(file)
            print(f_data.shape)
            
    print('Test success!')
