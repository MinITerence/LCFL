function V_E = VE(l)
global I noise Q tau Lmax Bmax delta G Pmax Emax
V_E = Q/l ./ log2( G .* Emax /l /noise + 1 );
end