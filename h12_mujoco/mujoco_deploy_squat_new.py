import pygame
from h12_controller_squat import H12_Controller_Squat

######################################################################
# Main simulation
def main():
    """Main function to run the H12 simulation."""
    # Initialize pygame for input handling
    pygame.init()
    pygame.display.set_mode((300, 100))
    
    # Create and run the controller with squat policy
    controller = H12_Controller_Squat("h1_2.yaml", policy_name="squat")
    controller.run_simulation()

######################################################################
# Entry point
if __name__ == "__main__":
    main()