"""
To gather frames of each trajectory to one folder
"""

import os
import re
import imageio.v2 as imageio
import pickle
import platform

def get_storage_dir():
    platform_info = platform.uname()
    system = platform_info[0]
    node = platform_info[1]
    if system.lower() == 'linux':
        if 'ohws59' in node:
            return "/local/home/xiazhi/Desktop/code/lunar-lander/data/"
        if 'eu' in node:
            return "/cluster/work/hilliges/xiazhi/lunar-lander/data/"
    return "/Users/anniezhi/Desktop/MasterThesis_RLwithUserIntention/code/lunar-lander/data/"


TRAIN_DIR = get_storage_dir()
ROWS = 64
COLS = 64
CHANNELS = 1

# generate filenames from the data folder
image_filenames = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if not i.startswith('.')] # use this for full dataset
key1 = lambda x: x.split('/')[-1][:25]
image_filenames.sort(key=key1)
key2 = lambda x: int(re.search(r'(?<=_)\d+(?=_\d+\.jpeg$)', x).group())
image_filenames.sort(key=key2)

# Iterate throuigh the filenames and for each one load the image, resize and normalise
actions_list = []
actions = []

episodes_list = []
episode = []
frame_id_last = 0
for i, image_file in enumerate(image_filenames):
    match_frame = re.search(r'(?<=_)\d+(?=_\d+\.jpeg$)', image_file)
    frame_id = int(match_frame.group())
    match_action = re.search(r"\d+(?=.jpeg)", image_file)
    action = int(match_action.group())

    print(len(episodes_list), '\t', image_file, '\t', frame_id)

    if frame_id < frame_id_last:
        episodes_list.append(episode)
        episode = [image_file]
        actions_list.append(actions)
        actions = [action]
    else:
        episode.append(image_file)
        actions.append(action)
    frame_id_last = frame_id

episodes_list.append(episode)
actions_list.append(actions)

# save image to gifs
for i, episode in enumerate(episodes_list):
    print(f'saving {i} / {len(episodes_list)}')
    images = []
    for frame in episode:
        images.append(imageio.imread(frame))
    if 'lunar-lander' in os.getcwd():
        imageio.mimsave(os.getcwd()+f'/data_new/expert/gifs/{i}.gif', images)
    else:
        imageio.mimsave(os.getcwd()+f'/lunar-lander/data_new/expert/gifs/{i}.gif', images)

# save actions
if 'lunar-lander' in os.getcwd():
    with open(os.getcwd()+f'/data_new/expert/actions.pkl', "wb") as fp:
        pickle.dump(actions_list, fp)
else:
    with open(os.getcwd()+f'/lunar-lander/data_new/expert/actions.pkl', "wb") as fp:
        pickle.dump(actions_list, fp)

## move data to different folders
# for filename, dest_folder in file_list:
#     # Construct the source path by joining the filename with the current working directory
#     src_path = os.path.join(os.getcwd(), filename)
#     # Construct the destination path by joining the destination folder with the filename
#     dest_path = os.path.join(dest_folder, filename)
#     # Move the file to the destination folder
#     shutil.move(src_path, dest_path)

# image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)        
# data[i] = image
# data[i] = data[i]/255
# if i%1000 == 0: print('Processed {} of {}'.format(i, count))

# Create a data array for image data
# count = len(image_filenames)
# data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.float)