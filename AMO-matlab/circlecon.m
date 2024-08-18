function [c,ceq] = circlecon(var)
global I noise Q tau Lmax Bmax delta G Pmax Emax

N = var(1);
p = var(2:I+1);
b = var(I+2:2*I+1);

m_1 = p * Q /N ./b ./log2( 1 + p .*G /noise ) - Emax;
m_2 = Q /N ./b ./log2( 1 + p .*G /noise ) + N^3 * tau - Lmax;
c1 = max(m_1);
c2 = max(m_2);
c = max(c1,c2);
ceq = [];
end