"""Tests for neuropaths.data.

TODO:
    * CommittorDataset: __len__ equals the number of unique solution_ids.
    * __getitem__(0) returns shapes ([C, G, G], [G, G]).
    * generate_dataset: writes a CSV whose columns match the schema
      (solution_id, x, y, b1, b2, rho[, finv]) and whose x-coordinates
      are the expected `np.linspace(0, 1, G)`.
    * Small smoke test with num_solutions=2 runs end-to-end in < 10s
      (gate with pytest.mark.slow if FD solve exceeds that).
"""
