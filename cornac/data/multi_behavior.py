import numpy as np
import scipy.sparse as sp
from cornac.data import Dataset


class PurchaseViewDataset(Dataset):
    """a multi-behavior dataset

    Parameters
    ----------
    dataset: cornac.data.Dataset
        The base Cornac dataset containing the primary interactions.

    view_matrix: scipy.sparse.csr_matrix
        The sparse matrix containing the secondary interactions.
        Its dimensions must match the primary dataset.
    """

    def __init__(self, dataset, view_matrix):
        # Inherit all properties (ID maps, num_users, num_items, matrix) from the base dataset
        self.__dict__.update(dataset.__dict__)
        self.view_matrix = view_matrix

    @classmethod
    def build_from_raw(cls, base_dataset, raw_view_data):
        """
        Parameters
        ----------
        base_dataset: cornac.data.Dataset
            The already-built base dataset containing the `uid_map` and `iid_map`.

        raw_view_data: list of tuples (user_id, item_id, rating/weight)
            The raw, unmapped view interactions.

        Returns
        -------
        PurchaseViewDataset
        """
        row_indices = []
        col_indices = []

        # synchronize IDs: only keep view interactions for users/items that exist in the base dataset
        for raw_u, raw_i, _ in raw_view_data:
            if raw_u in base_dataset.uid_map and raw_i in base_dataset.iid_map:
                row_indices.append(base_dataset.uid_map[raw_u])
                col_indices.append(base_dataset.iid_map[raw_i])

        # build the sparse CSR matrix for views
        view_matrix = sp.csr_matrix(
            (np.ones(len(row_indices)), (row_indices, col_indices)),
            shape=(base_dataset.num_users, base_dataset.num_items)
        )

        return cls(base_dataset, view_matrix)