from utils import clear_screen
from modules.dataset import dataset_module
from modules.models import model_module
from modules.xai import xai_module

def prompt():
    clear_screen() #clears the terminal screen
    #prompts the user for 3 modules
    print("Welcome to XAI Using Texture Analysis")
    print("1. Models")
    print("2. XAI")
    print("3. Exit")

    #get input from user
    choice = None
    while True:
        try:
            choice = int(input("Which of the available modules would you like to access (1-3): "))
            if 1 <= choice <= 3:
                break  # Exit the loop if the input is valid
            else:
                print("Please choose one of the 3 available modules.")
        except ValueError:
            print("That's not an integer. Please try again.")

    return choice

def main():
    #Execute the main program
    print("main")

    choice = prompt()
    # choice=1

    while choice != 3:

        #select the chosen module
        if choice == 1:
            #dataset_module()
            model_module()
        
        elif choice == 2:
            xai_module()

        choice = prompt()

if __name__ == '__main__':
    main()