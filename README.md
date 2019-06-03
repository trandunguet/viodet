# Requirement
Anaconda:

- python 2.7
- bob.ip.optflow.liu
- keras

Pip:

- opencv-python
- scikit-learn

# Get BEHAVE dataset
`$ bash script/get_BEHAVE_dataset.sh`

# Cut video into segments
`$ python cut_video.py VIDEO_PATH MARKUP_FILE_PATH OUTPUT_DIR`

# Extract video segments to vif
`$ python extract_feature.py INPUT_FOLDER OUTPUT_FOLDER`

# Train
`$ python train.py INPUT_1 ... INPUT_N OUTPUT_FOLDER`

# Test
`$ python test.py TEST_SET MODEL_FOLDER`
