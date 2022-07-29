import gdown

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

def get_model(output_path):
    '''
    downloads a model from shared google drive and saves it in 'output_path'
    
    Arguments:
        - output_path [str] - where the model should be saved
    Outputs:
        - model [pth] saved in the output path directory
    '''

    url = 'https://drive.google.com/uc?id=1uHNADViICyJEjJljv747nfvrGu12kjtu'
    gdown.download(url, output_path, quiet=False)

if __name__ == '__main__':
    get_model('trained_models/affecnet8.pth')


    