"""LateGraph: repairing temporal-GNN memory when events arrive late, get amended, or must be deleted.

Main pieces
  data.EventStream            one timestamped interaction stream (src, dst, t, msg) + chronological split
  delays                      delay models that give every test event an arrival time (the second clock)
  model                       the PyG TGN we study (TGNMemory + TransformerConv embedding + MLP decoder)
  memory.RetroactiveTGNMemory a TGNMemory subclass that can insert, delete and amend past events and
                              repair its memory exactly (dependency cone) or approximately (epsilon-repair)
  neighbors.TimeOrderedNeighborStore  last-k neighbours ordered by occurrence time (insertion-order invariant)
  stream                      streaming evaluation of repair policies under late arrivals
  attacks                     membership tests for deleted interactions
  contraction                 Jacobian norms of the memory update and the epsilon-repair certificate
"""
__version__ = "0.2.0"
