
tic
clear
I = 20; % client number
noise  = 10e-9; 
Q = 1e8; %model size
tau = 1e-4; 
Lmax = 5; %max latency
Bmax = 4e8; %bandwidth
delta = 1e-3; %error

% channel 
rng(1); % seed
Gmagnitude = 1e-9;
lowerRange = 1 * Gmagnitude; 
upperRange = 10 * Gmagnitude;  
G = lowerRange + (upperRange-lowerRange).*rand(1, I); %channel power gain

% transmit power
rng(2);  
Pmagnitude = 0.1;
lowerRange = 5 * Pmagnitude;  
upperRange = 10 * Pmagnitude;  
Pmax = lowerRange + (upperRange-lowerRange).*rand(1, I);  

%energy
rng(3); 
Pmagnitude = 1;
lowerRange = 5 * Pmagnitude;  
upperRange = 10 * Pmagnitude;  
Emax = lowerRange + (upperRange-lowerRange).*rand(1, I); 


parfor k = 1:20000

rng("shuffle")
p = Pmax.*rand(1, I); 

rng("shuffle")
r = rand(1,I);
b = r/sum(r).*Bmax;

rng("shuffle")
N = ceil( 1 + (I-1).*rand(1) );


m_1 = p * Q /N ./b ./log2( 1 + p .*G /noise ) - Emax;
m_2 = Q /N ./b ./log2( 1 + p .*G /noise ) + N^3 * tau - Lmax;

if (max(m_1)<=0 && max(m_2)<=0)
    N_star(k) = N;
else
    N_star(k) = inf;
end

end
toc

min(N_star)


