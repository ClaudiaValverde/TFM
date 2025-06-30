import os
import torch
import torch.nn as nn
from tqdm import tqdm



d_model = 256

# Define a linear layer for dimensionality reduction
class DimReducer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DimReducer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)



def reduction(input_dir, d_model, output_dir):

    # Create output directory if it doesn't exist
    #os.makedirs(output_dir, exist_ok=True)

    # Load the dictionary of embeddings
    weights = torch.load(input_dir, weights_only=False)
    weights = {key.rstrip(): value for key, value in tqdm(weights.items(), desc='Loading embeddings')} #perq em donava error al la seq acabar en '\n'

    # Determine input dimension from the first embedding
    first_tensor = next(iter(weights.values()))
    input_dim = first_tensor.shape[-1]
    print('input_dim: ', input_dim)

    reducer = DimReducer(input_dim, d_model)
    
    # Apply reduction to all embeddings
    '''
    reduced_weights = {key: reducer(value) for key, value in tqdm(weights.items(), desc='Reducing embeddings')}
    print('Number of embeddings:', len(reduced_weights))
    '''
    reducer.eval()

    with torch.no_grad():

        reduced_weights = {}
        for key, value in tqdm(weights.items(), desc='Reducing embeddings'):
            print(f"Processing key: {key}")
            print(f"Original value shape: {value.shape}")

            tensor_value = torch.tensor(value, dtype=torch.float32, device='cpu')

            reduced_value = reducer(tensor_value)
            print(f"Reduced value shape: {reduced_value.shape}")
            reduced_weights[key] = reduced_value
        
        print('Number of embeddings:', len(reduced_weights))
        # Save the reduced embeddings
        torch.save(reduced_weights, output_dir)

        print("Embeddings reduced and saved.")

'''
print(' ===== Reduction of ESM2 embeddings: =====')

# load embeddings
input_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/plinder_prots_esm_nochain_1280.pt'
output_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/plinder_prots_esm_nochain_256_reduced_good.pt'

reduction(input_dir, d_model, output_dir)

print(' ===== Reduction of PickPocket embeddings: =====')

# load embeddings
input_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/plinder_embeddings_pocket_more_4352.pt'
output_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/plinder_embeddings_pocket_more_256_reduced_good.pt'

reduction(input_dir, d_model, output_dir)

print(' ===== Reduction of SaProt embeddings: =====')

# load embeddings
input_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/saprot_plinder.pt'
output_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/saprot_plinder_256_reduced_good.pt'

reduction(input_dir, d_model, output_dir)


print(' ===== Reduction of TensorDTI lastlayer embeddings: =====')

# load embeddings
input_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/post_tensor/triple_hidden_512.pt'
output_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/triple_hidden_256_reduced_good.pt'

reduction(input_dir, d_model, output_dir)
'''

print(' ===== Reduction of RETRO Pocket embeddings: =====')

# load embeddings
input_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/retro/retros_embeddings_pocket_all.pt'
output_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/retros_embeddings_pocket_all_good.pt'

reduction(input_dir, d_model, output_dir)

print(' ===== Reduction of RETRO ESM2 embeddings: =====')

# load embeddings
input_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/retro/retros_prots_esm_mean1280.pt'
output_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/retros_prots_esm_mean1280_good.pt'

reduction(input_dir, d_model, output_dir)

print(' ===== Reduction of RETRO SaProt embeddings: =====')

# load embeddings
input_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/retro/prots_retro.pt'
output_dir = '/gpfs/projects/nost02/Prot2Drug/plinder/pocket/prots_retro_good.pt'

reduction(input_dir, d_model, output_dir)