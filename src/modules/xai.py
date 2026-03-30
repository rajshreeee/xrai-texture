from utils import clear_screen
from xai.laplacian import analyze_laplacian
from xai.GLCM import analyse_GLCM
from xai.GLCM_Unet import analyze_GLCM_Unet
from xai.GLCM_Unet import analyze_GLCM_Hrnet
from xai.LTEM_Unet import LTEM_analysis_unet
from xai.LTEM_hrnet import LTEM_analysis_hrnet
import config

from xai.LTEM_Cosine_Similarity import cosine_similarity_analysis as cosine_similarity_analysis_ltem
from xai.LTEM_StackedEnsemble_Cosine_Similarity import cosine_similarity_analysis 

def prompt_analysis():
    print("1. GLCM feature Map Analysis")
    print("2. LTEM feature Map Cosine Similarity Analysis")
    print("3. LTEM & Stacked Ensemble feature Map Cosine Similarity Analysis")
    print("4. Laplacian Feature Analysis")
    print("5. Exit")

    choice = None
    while True:
        try:
            choice = int(input("Select Function (1-5): "))
            if 1 <= choice <= 5:
                break
            else:
                print("Please choose one of the 5 available functions.")
        except ValueError:
            print("That's not an integer. Please try again.")

    return choice

def prompt_model():
    clear_screen()

    print("1. DeeplapV3")
    print("2. FCN")
    print("3. U-Net")
    print("4. HR-Net")
    print("5. FPN-Net")
    print("6. Link-Net")

    choice_model = None
    while True:
        try:
            choice_model = int(input("Select Model (1-6): "))
            if 1 <= choice_model <= 6:
                break
            else:
                print("Please choose one of the 6 available models.")
        except ValueError:
            print("That's not an integer. Please try again.")

    return choice_model

def prompt_dataset():
    print("1. CBIS_DDSM")
    print("2. CBIS_DDSM_CLAHE")
    print("3. HAM10000")
    print("4. HAM10000_CLAHE")
    print("5. POLYP")
    print("6. POLYP_CLAHE")

    choice_dataset = None
    while True:
        try:
            choice_dataset = int(input("Select Dataset (1-6): "))
            if 1 <= choice_dataset <= 6:
                break
            else:
                print("Please choose one of the 6 available datasets.")
        except ValueError:
            print("That's not an integer. Please try again.")

    return choice_dataset

def xai_module():
    clear_screen()
    print("Welcome to the Texture Analysis Module")

    while True:
        choice_analysis = prompt_analysis()

        if choice_analysis == 5:
            print("Exiting the Texture Analysis Module. Goodbye!")
            break

        elif choice_analysis == 4:
            print("\nRunning Laplacian Feature Analysis...\n")
            analyze_laplacian()

        elif choice_analysis == 3:
            print("\nRunning LTEM & Stacked Ensemble Cosine Similarity Analysis...\n")
            test_image_path = config.CBIS_DDSM_dataset_path + '/test/images/Mass-Training_P_00133_LEFT_CC_crop7.jpg'
            texture_image_paths = [config.CBIS_DDSM_dataset_path + f'/test/textures/Feature_{i}/Mass-Training_P_00133_LEFT_CC_crop7.jpg' for i in range(1, 10)]
            csv_result_path = config.results_path + '/LTEM/StackedEnsemble/CBIS_DDSM/CBIS_DDSM_StackedEnsemble_LTEM.csv'
            png_result_path = config.results_path + '/LTEM/StackedEnsemble/CBIS_DDSM/CBIS_DDSM_StackedEnsemble_LTEM.png'
            cosine_similarity_analysis(test_image_path, texture_image_paths, csv_result_path, png_result_path)

        else:
            choice_model = prompt_model()
            choice_dataset = prompt_dataset()

            if choice_analysis == 1:
                print("\nRunning GLCM Feature Map Analysis...\n")
                if choice_model == 3:
                    analyze_GLCM_Unet(choice_dataset)
                elif choice_model == 4:
                    analyze_GLCM_Hrnet(choice_dataset)
                else:
                    analyse_GLCM(choice_model, choice_dataset)

            elif choice_analysis == 2:
                print("\nRunning LTEM Cosine Similarity Analysis...\n")
                if choice_model == 3:
                    LTEM_analysis_unet(choice_dataset)
                elif choice_model == 4:
                    LTEM_analysis_hrnet(choice_dataset)
                else:
                    cosine_similarity_analysis_ltem(choice_model, choice_dataset)
