import numpy as np
import matplotlib.pyplot as plt


A = np.array([
    [3, 2, 2],
    [2, 3, -2]
])

print("Original matrix A:\n", A)

# Thực hiện SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)
print("\nU matrix:\n", U)
print("\nSingular values (S):\n", S)
print("\nVT matrix:\n", VT)

# Plot singular values
plt.plot(S, 'o-')
plt.title("Singular Values")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

# Chọn số chiều k muốn giữ lại (ví dụ: k=1)
k = 1
U_k = U[:, :k]
S_k = np.diag(S[:k])
VT_k = VT[:k, :]
A_k = np.dot(U_k, np.dot(S_k, VT_k))

print(f"\nReconstructed matrix A_k (k={k}):\n", np.round(A_k, 3))

# Đánh giá sai số tái tạo
rmse = np.sqrt(np.mean((A - A_k) ** 2))
print(f"\nRMSE between original A and reconstructed A_k: {rmse:.4f}")

# Có thể xuất A_k ra file nếu muốn
np.savetxt(f"reconstructed_k{k}.csv", A_k, delimiter=",", fmt="%.3f")
