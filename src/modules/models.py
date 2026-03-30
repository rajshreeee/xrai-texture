from utils import clear_screen
from models.train import train_model
from models.test import test_model

def prompt():
    print("1. Train Model")
    print("2. Test Model")
    print("3. Exit")

    choice = None
    # choice =2
    while True:
            try:
                choice = int(input("Select Function (1-3): "))
                if 1 <= choice <= 3:
                    break  # Exit the loop if the input is valid
                else:
                    print("Please choose one of the 3 available functions.")
            except ValueError:
                print("That's not an integer. Please try again.")
    print("choice")
    return choice

def model_module():
    print("Welcome to the Models Module")
    
    #get input from user
    choice = prompt()
    # choice = 1

    #perform user requested tasks
    while choice != 3:
        
        if choice == 1:
            train_model()

        
        elif choice == 2:
            test_model()
            
        choice = prompt()

    return
