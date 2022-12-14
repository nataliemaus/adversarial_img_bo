import numpy as np
import torch 


class Objective:
    '''Base class for any optimization task
        class supports oracle calls and tracks
        the total number of oracle class made during 
        optimization 
    ''' 
    def __init__(
        self,
        num_calls=0,
        task_id='',
        dim=256,
        lb=None,
        ub=None,
    ):
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # string id for optimization task, often used by oracle
        #   to differentiate between similar tasks (ie for guacamol)
        self.task_id = task_id
        # latent dimension of vaae
        self.dim = dim
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub  

    def __call__(self, xs):
        ''' Input 
                x: a numpy array or pytorch tensor of search space points
            Output
                out_dict['valid_xs'] = an array of xs passed in from which valid scores were obtained 
                out_dict['scores']: an array of valid scores obtained from input xs
        '''
        # if type(xs) is np.ndarray: 
        #     xs = torch.from_numpy(xs).float()
        # out_dict = self.xs_to_valid_scores(xs)
        # return out_dict
        self.num_calls += xs.shape[0]
        return self.query_oracle(xs)

    # def xs_to_valid_scores(self, xs):
    #     scores = []
    #     for x in xs:
    #         score = self.query_oracle(x)
    #         # track number of oracle calls 
    #         #   nan scores happen when we pass an invalid
    #         #   molecular string and thus avoid calling the
    #         #   oracle entirely
    #         if np.logical_not(np.isnan(score)):
    #             self.num_calls += 1
    #         scores.append(score)
    #     scores_arr = np.array(scores)
    #     if type(xs) is list: 
    #         xs = np.array(xs) 
    #     elif type(xs) is torch.Tensor:
    #         xs = xs.detach().cpu().numpy() 
    #     # get valid zs, xs, and scores
    #     bool_arr = np.logical_not(np.isnan(scores_arr)) 
    #     valid_xs = xs[bool_arr]
    #     valid_scores = scores_arr[bool_arr]
    #     out_dict = {}
    #     out_dict['scores'] = valid_scores
    #     out_dict['bool_arr'] = bool_arr
    #     out_dict['valid_xs'] = valid_xs 

    #     return out_dict


    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        raise NotImplementedError("Must implement query_oracle() specific to desired optimization task")
