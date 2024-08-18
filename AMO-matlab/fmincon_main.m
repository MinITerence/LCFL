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
fun = @(var) var(1);  %var = [N,p,b]
lb = zeros(1,2*I+1);    
ub = [I,Pmax,Bmax.*ones(1,I)]; 
A = [0,zeros(1,I),ones(1,I)];       
Aeq = []; 
beq = [];  
x0 = [I/2,Pmax/2,Bmax.*ones(1,I)/I];   
nonlcon = @circlecon;
% interior-point trust-region-reflective sqp sqp-legacy active-set 

options = optimoptions('fmincon','Algorithm','interior-point');  
var = fmincon(fun,x0,A,Bmax,Aeq,beq,lb,ub,nonlcon,options); 

N_star = ceil(var(1))
toc
