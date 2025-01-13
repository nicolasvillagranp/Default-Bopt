from src_torch.resnet_tune import evaluate
from skopt import gp_minimize
from skopt.space import Integer, Real
from src_torch.utils import set_seed
from tqdm import tqdm 




def evaluate_wrapped(params):
    lr_model, lr_linear, dropout, hidden_size, batch_size = params
    return evaluate(lr_model, lr_linear, dropout, int(hidden_size), int(batch_size))

SEED = 42
set_seed(SEED)


def print_step(res, progress_bar):
    progress_bar.set_postfix({'Best Value': res.fun})
    progress_bar.update(1)  # Update the progress bar by one step


if __name__ == '__main__':
    dimensions = [Real(1e-5, 1e-2, name='lr_model'), 
                Real(1e-4, 1e-2, name='lr_linear'),
                Real(0.0, 0.7, name='dropout'), 
                Integer(32, 1024, name='hidden_size'),
                Integer(32, 512, name='batch_size')]

    init_params = (1e-4, 1e-3,0.2, 64, 128)
    y = evaluate(*init_params) 

    n_calls = 20
    total_calls = n_calls + 1

    print('---------------- Default Optimization -----------------------------')
    # Initialize tqdm progress bar
    with tqdm(total=total_calls, desc="Optimizing", dynamic_ncols=True) as progress_bar:
        # Run the optimizer with the callback
        optimizer = gp_minimize(
            evaluate_wrapped,                        
            dimensions=dimensions,         
            acq_func="EI",                    
            n_calls=total_calls,              
            n_initial_points=0,               
            x0=[init_params],                 
            y0=[y],                           
            random_state=SEED,                  
            callback=[lambda res: print_step(res, progress_bar)]  
        )


    print('---------------- Random Optimization -----------------------------')
    with tqdm(total=total_calls, desc="Optimizing", dynamic_ncols=True) as progress_bar:
        # Run the optimizer with the callback
        optimizer = gp_minimize(
            evaluate_wrapped,                        
            dimensions=dimensions,         
            acq_func="EI",                    
            n_calls=n_calls,              
            n_initial_points=1,                                        
            random_state=SEED,                  
            callback=[lambda res: print_step(res, progress_bar)]  
        )




