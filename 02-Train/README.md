How to create the different input representations:

**ESM2:** https://github.com/facebookresearch/esm \
**SaPROT:** https://github.com/westlake-repl/SaProt \
**PickPocket:** embeddings obtained from in-house model, 4352 dimension \
**Tensor-DTI:** embeddings obtained from in-house model, 256 dimension 

For those obtained representations that are not 256, use `dummy_reduction.py` to reduce the dimension with a dummy linear layer.

Once the different represntations have been preprocessed, they can be input to the training script `main_code.py`, always one type of protein representation either 1) ESM2, 2) SaProt or 3) TendorDTI protein and/or one type of pocket representation 1) PickPocket or 2) TensorDTI pocket.

The smiles representations have been preprocessed in the previous folder.
