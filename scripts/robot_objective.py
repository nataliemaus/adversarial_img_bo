import torch 
import sys 
sys.path.append("../../")
from robot.robot.objective import Objective

### ... 
# NOTE: ROBOT is NOT set up well to handle args other than tau
# being passed into adversarial objective... 
# so do some more exploration to determine final best default settings 
# for some args, and then find way to allow passing in of other args... 

class AdversarialsObjectiveRobot(Objective):
    ''' Adv Img Optimization task
        Goal: Minimize loss of desired img class
    ''' 
    def __init__(
        self,
        xs_to_scores_dict={},
        num_calls=0,
        tau=None,
        adv_obj_object=None,
        **kwargs,
    ):
        self.adv_obj_object = adv_obj_object

        super().__init__(
            xs_to_scores_dict=xs_to_scores_dict,
            num_calls=num_calls,
            task_id='advImg',
            dim=adv_obj_object.dim,
            lb=adv_obj_object.lb,
            ub=adv_obj_object.ub,
            **kwargs,
        ) 


    def query_oracle(self, x):
        prompts_next, y_next = self.adv_obj_object(x)
        return y_next


    def divf(self, x1, x2 ):
        toks1 = self.adv_obj_object.proj_word_embedding(x1)
        toks2 = self.adv_obj_object.proj_word_embedding(x2)

        if len(toks1.shape) == 0:
            toks1 = toks1.unsqueeze(0)
        if len(toks2.shape) == 0:
            toks2 = toks2.unsqueeze(0)

        # Distance = how many tokens are different: 
        return torch.cdist(toks1, toks2, p=0.0)
