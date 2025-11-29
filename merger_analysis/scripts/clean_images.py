import os
import shutil
import pickle
import glob


def check_file_validity (pickle_name):
    with open(pickle_name, 'rb') as f:
        data = pickle.load(f)
        
    masked_fraction = (data['mask']>0).sum() / data['mask'].size
    if masked_fraction > 0.2:
        return False
        
    return True
    
    
def main (output_dir):
    files = glob.glob(f'{output_dir}/M*/*i_results.pkl')
    
    if not os.path.exists(f'{output_dir}/invalid_images/'):
        os.mkdir(f'{output_dir}/invalid_images/')
    
    for file in files:
        merian_id = os.path.basename(file).split('_')[0]
        dirname = os.path.dirname(file)
        is_valid = check_file_validity(file)
        if not is_valid:
            shutil.move(dirname, f'{output_dir}/invalid_images/{merian_id}')
            print(f'Flagged {merian_id} as an invalid image.')

if __name__ == '__main__':
    main('../../local_data/pieridae_output/starlet/msorabove_v1/')