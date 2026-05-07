import torch

class CliffordAlgebra:
    """
    Cl(3,0) algebra utilities (Euclidean 3D).

    Multivector basis (8 components) ordering:
      0: 1
      1: e1, 2: e2, 3: e3
      4: e12, 5: e23, 6: e31
      7: e123

    Representation: tensor [..., C, 8], where C is the feature channel.
    """
    def __init__(self, device=None):
        self.device = device
        self._cayley = None

    def to(self, device):
        self.device = device
        self._cayley = None
        return self

    @property
    def cayley(self):
        if self._cayley is None:
            self._cayley = self._build_cayley().to(self.device) if self.device is not None else self._build_cayley()
        return self._cayley

    def _build_cayley(self):
        """
        Build Cayley table C (8x8x8) such that:
            basis[i] * basis[j] = sum_k C[i,j,k] * basis[k]
        In Cl(3,0), every basis blade product maps to a single blade with coefficient ±1.
        """
        # Bitmask for each basis blade in our ordering.
        # Masks use canonical bit positions: e1=1, e2=2, e3=4.
        masks = [0, 1, 2, 4, 3, 6, 5, 7]  # 1, e1, e2, e3, e12, e23, e31, e123

        # Our basis differs from canonical ascending blades only for e31:
        # mask 5 corresponds to canonical e13 (e1^e3). Our basis uses e31 = e3e1 = -e13.
        basis_sign = [1, 1, 1, 1, 1, 1, -1, 1]

        mask_to_idx = {m: i for i, m in enumerate(masks)}
        metric = [1, 1, 1]  # Euclidean: e_i^2 = +1

        def gp_sign(a_mask: int, b_mask: int) -> float:
            # Sign from reordering (anti-commutation) + metric for repeated indices.
            sign = 1.0
            for i in range(3):
                if (a_mask >> i) & 1:
                    below = b_mask & ((1 << i) - 1)
                    if (below.bit_count() & 1) == 1:
                        sign *= -1.0
            common = a_mask & b_mask
            for i in range(3):
                if (common >> i) & 1:
                    sign *= float(metric[i])
            return sign

        C = torch.zeros(8, 8, 8, dtype=torch.float32)
        for i, mi in enumerate(masks):
            for j, mj in enumerate(masks):
                # Our basis element = basis_sign[i] * canonical_blade(mi)
                sg = gp_sign(mi, mj)
                mr = mi ^ mj
                k = mask_to_idx[mr]
                # Convert canonical result back into our basis (divide by basis_sign[k])
                coeff = basis_sign[i] * basis_sign[j] * sg / basis_sign[k]
                C[i, j, k] = float(coeff)
        return C

    # ---------- representation helpers ----------
    def split_grades(self, x):
        s = x[..., 0:1]   # [..., C, 1]
        v = x[..., 1:4]   # [..., C, 3]
        b = x[..., 4:7]   # [..., C, 3]
        t = x[..., 7:8]   # [..., C, 1]
        return s, v, b, t

    def cat_grades(self, s, v, b, t):
        return torch.cat([s, v, b, t], dim=-1)

    def vector_to_mv(self, edge_vector):
        """
        edge_vector: [..., C, 3] -> mv [..., C, 8]
        Put edge_vector into the vector grade (e1,e2,e3) with all other grades = 0.
        """
        assert edge_vector.size(-1) == 3, f"edge_vector last dim must be 3, got {edge_vector.shape}"

        zeros_s = torch.zeros(edge_vector.shape[:-1] + (1,),
                            device=edge_vector.device, dtype=edge_vector.dtype)  # scalar
        zeros_b = torch.zeros(edge_vector.shape[:-1] + (3,),
                            device=edge_vector.device, dtype=edge_vector.dtype)  # bivector (e12,e23,e31)
        zeros_t = torch.zeros(edge_vector.shape[:-1] + (1,),
                            device=edge_vector.device, dtype=edge_vector.dtype)  # pseudoscalar (e123)

        # [s(1), v(3), b(3), t(1)] => total 8
        return torch.cat([zeros_s, edge_vector, zeros_b, zeros_t], dim=-1)

    # ---------- products ----------
    def geometric_product(self, a, b):
        C = self.cayley.to(a.device)
        return torch.einsum('...ci,...cj,ijk->...ck', a, b, C)

    def geometric_product_vector_edge(self, a, edge_vector):
        return self.geometric_product(a, self.vector_to_mv(edge_vector))

    def geometric_product_vector_edge_left(self, edge_vector, b):
        return self.geometric_product(self.vector_to_mv(edge_vector), b)
