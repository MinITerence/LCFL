clear

global I noise Q tau Lmax Bmax delta G Pmax Emax
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

tic
l1 = delta;
l2 = Lmax;
l1_hat = l1 + 0.382*(l2-l1);
l2_hat = l1 + 0.618*(l2-l1);
h1 = f(l1_hat);
h2 = f(l2_hat);

% Compute the minimal point
while l2-l1 > delta
    if h1 > h2 
        l1 = l1_hat;
        l1_hat = l2_hat;
        h1 = h2;
        l2_hat = l1 + 0.618 * (l2 -l1);
        h2 = f(l2_hat);
    else
        l2 = l2_hat;
        l2_hat = l1_hat;
        h2 = h1;
        l1_hat = l1 + 0.382 * (l2 -l1);
        h1 = f(l1_hat);
    end
end


l_min = (l1+l2)/2;
if f(l_min) > 0 
    disp("Infeasible1")
    return
end


% find the maximum l
l3 = l_min;
l4 = Lmax;
while l4 - l3 > delta
    l_hat = (l3 + l4)/2;
    h_hat = f(l_hat);
    if h_hat < 0
        l3 = l_hat;
    else
        l4 = l_hat;
    end
end

N = sum( max( VE(l_hat), VP(l_hat) ) ) / Bmax;
N_star = ceil(N);
if N_star > I
    disp("Infeasible2")
end

b = max( VE(l_hat), VP(l_hat) ) * Bmax / sum( max( VE(l_hat), VP(l_hat) ) );
p = 1 ./ G .* ( 2.^( Q / l_hat ./b /N_star) -1) * noise;
toc



