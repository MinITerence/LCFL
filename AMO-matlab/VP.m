function V_P = VP(l)
global I noise Q tau Lmax Bmax delta G Pmax Emax
V_P = Q/l ./ log2( G .* Pmax /noise + 1 );
end