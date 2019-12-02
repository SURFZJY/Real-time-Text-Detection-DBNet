import os

'''
generate the txt file for train/test in our framework
'''

############ Utility functions ############
def get_images(img_path):
    '''
    find image files in data path
    :return: list of files found
    '''
    img_path = os.path.abspath(img_path)
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG', 'PNG']
    for parent, dirnames, filenames in os.walk(img_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return sorted(files)
    
def get_txts(txt_path):
    '''
    find gt files in data path
    :return: list of files found
    '''
    txt_path = os.path.abspath(txt_path)
    files = []
    exts = ['txt']
    for parent, dirnames, filenames in os.walk(txt_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} txts'.format(len(files)))
    return sorted(files)    

if __name__ == '__main__':
    img_path = '/home1/surfzjy/data/ICDAR-13and15/train_data'
    files = get_images(img_path)
    txt_path = '/home1/surfzjy/data/ICDAR-13and15/train_data'
    txts = get_txts(txt_path)
    n = len(files)
    assert len(files) == len(txts)
    with open('dataset_ic13and15_train.txt', 'w') as f:
        for i in range(n):
            line = files[i] + '\t' + txts[i] + '\n'
            f.write(line)
    
    print('dataset generated ^_^ ')