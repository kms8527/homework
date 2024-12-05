import numpy as np

# Define the vectors for part (a)
vectors_a = np.array([[1, 0, 1], [0, 1, 1], [1, 2, 3]], dtype=float)

# Gram-Schmidt process
def gram_schmidt(vectors):
    def projection(u, v):
        return (np.dot(v, u) / np.dot(u, u)) * u

    orthogonal = []
    for v in vectors:
        for u in orthogonal:
            v -= projection(u, v)
        orthogonal.append(v)

    # Normalize the vectors to get orthonormal basis
    orthonormal = [vec / np.linalg.norm(vec) for vec in orthogonal]
    return np.array(orthonormal)

# Apply the Gram-Schmidt process to the vectors of part (a)
# Define the vectors for part (b)
vectors_b = np.array([[1, -2, -1, 3], [3, 6, 3, -1], [1, 4, 2, 8]], dtype=float)

# Apply the Gram-Schmidt process to the vectors of part (b)
orthonormal_basis_b = gram_schmidt(vectors_b)
print(orthonormal_basis_b)