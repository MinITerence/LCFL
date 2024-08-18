function h = f(l)
global I noise Q tau Lmax Bmax delta G Pmax Emax
V_E = VE(l);
V_P = VP(l);
m = max(V_E, V_P);
s = sum(m);
h = s/Bmax - ((Lmax - l)/tau)^(1/3);
end