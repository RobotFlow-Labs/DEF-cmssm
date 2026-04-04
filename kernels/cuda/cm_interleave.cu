// DEF-cmssm scaffold
// TODO: implement cross-modal interleave/de-interleave CUDA kernels.
// Target contract:
//   interleave(rgb[B,D,L], thermal[B,D,L]) -> y[B,D,2L]
//   deinterleave(y[B,D,2L]) -> rgb[B,D,L], thermal[B,D,L]
