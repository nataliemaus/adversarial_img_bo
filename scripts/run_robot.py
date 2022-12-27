import fire 
import sys 
sys.path.append("../../")
sys.path.append("../")
from robot.scripts.continuous_space_optimization import ContinuousSpaceOptimization
from utils.robot_objective import AdversarialsObjectiveRobot

task_id_to_objective = {}
task_id_to_objective['advImg'] = AdversarialsObjectiveRobot

## edits needed to initialize AdversarialsObjectiveRobot 
#   w/ AdversarialsObjective object w/ correct args... 
#   Will do after finally figure out which version of opt works best... 
if __name__ == "__main__":
    fire.Fire(ContinuousSpaceOptimization)