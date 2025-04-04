To be verified!
NIS is using torchgeometry, which is using torch.gesv function in imgwarp.py. This part must be changed with: X = torch.linalg.solve(A, b)