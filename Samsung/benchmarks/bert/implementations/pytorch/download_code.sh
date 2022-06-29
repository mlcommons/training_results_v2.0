git clone https://github.com/SAITPublic/MLPerf_Training_v2.0 samsung_training_v2.0

# Copy samsung licensed source codes which do not exist in current directory  
cp samsung_training_v2.0/run_pretraining.py .
cp samsung_training_v2.0/utils.py .
cp samsung_training_v2.0/cleanup_scripts/run_split_and_chop_hdf5_files.py cleanup_scripts/

rm -rf samsung_training_v2.0



