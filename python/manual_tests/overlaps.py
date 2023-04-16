from time import monotonic
import ptens

H = ptens.modules.generate_generic_shape("path",5)
for i in range(5,10+3):
  n = 2**i
  G = ptens.modules.generate_generic_shape("path",n)
  x = G.nhoods(1)
  y = G.subgraphs(H)
  t0 = monotonic()
  ptens.graph.overlaps(x,y)
  t1 = monotonic()
  print(f"n {n}: {t1 - t0}(s)")

