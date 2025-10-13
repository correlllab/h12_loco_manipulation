import pygame
from h12_controller_walk import H12_Controller_Walk

######################################################################
# Main simulation
def main():
    """Main function to run the H12 walk simulation."""
    # Initialize pygame for input handling
    pygame.init()
    pygame.display.set_mode((300, 100))
    
    # Create and run the walk controller
    controller = H12_Controller_Walk("h1_2.yaml")
    controller.run_simulation()

######################################################################
# Entry point
if __name__ == "__main__":
    main()
