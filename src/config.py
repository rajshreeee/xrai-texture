import os

zip_url = "https://zenodo.org/records/10960991/files/CBIS-DDSM-Patches.zip?download=1"

#necessary path variables
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/data"
CBIS_DDSM_dataset_path = data_path + "/CBIS_DDSM"
CBIS_DDSM_CLAHE_dataset_path = data_path + "/CBIS_DDSM_CLAHE"
CBIS_DDSM_PATCHES=data_path+"/CBIS_DDSM_Patches_Mass_Context"
CBIS_DDSM_LAPLACIAN=data_path+"/CBIS_DDSM_LAPLACIAN"

HAM_dataset_path = data_path + "/HAM10000"
HAM_CLAHE_dataset_path = data_path + "/HAM10000_CLAHE"

POLYP_dataset_path = data_path + "/POLYP"
POLYP_CLAHE_dataset_path = data_path + "/POLYP_CLAHE"

#TEM_dataset_path = data_path + "/olddataset_traintestfolder_mask255_MassTrainingonly_pragati"
#patch_zip_path = data_path + "/CBIS-DDSM-Patches.zip"

maskdataset_path_mmsegmentation = "../data/SegmentationClass"
unet_config_file_path = "models/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py"
hrnet_config_file_path = "models/fcn_hr18_4xb2-160k_cityscapes-512x1024.py"

saved_models_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/saved_models"
results_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/results"