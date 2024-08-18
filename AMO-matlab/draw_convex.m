clear
global I noise Q tau Lmax Bmax delta G Pmax Emax
I = 20; % client number
noise  = 10e-9; 
Q = 2.2e8; %model size
tau = 1e-4;
Lmax = 5; %max latency
Bmax = 4e8; % total bandwidth
delta = 1e-2; % error

% channel initialization
rng(1); % seed
Gmagnitude = 1e-9;
lowerRange = 1 * Gmagnitude; 
upperRange = 10 * Gmagnitude; 
G = lowerRange + (upperRange-lowerRange).*rand(1, I); 

% max power
rng(2);    
Pmagnitude = 0.1;
lowerRange = 5 * Pmagnitude; 
upperRange = 10 * Pmagnitude;  
Pmax = lowerRange + (upperRange-lowerRange).*rand(1, I); 

% max energy
rng(3); 
Pmagnitude = 1;
lowerRange = 5 * Pmagnitude; 
upperRange = 10 * Pmagnitude; 
Emax = lowerRange + (upperRange-lowerRange).*rand(1, I); 

for k = 1:4
Q = 0.5e8 * k; % model size vary
l1 = 0.9;  %From delta, the figure is UGLY
l2 = Lmax;  

%test the montonicity
Index = floor((l2-l1)/delta);
for i = 1:Index
    x(k,i) = l1+delta*i;
    y(k,i) = f(i*delta+l1);
end
end
figure (1)
plot(x(1,:),y(1,:),'-',"Color","#f65314","LineWidth",2)
hold on
plot(x(2,:),y(2,:),'--',"Color","#7cbb00","LineWidth",2)
hold on
plot(x(3,:),y(3,:),'-.',"Color","#00a1f1","LineWidth",2)
hold on
plot(x(4,:),y(4,:),'.',"Color","#ffbb00","LineWidth",2)
hold on

legend('Q = 50 Mbits','Q = 100 Mbits','Q = 150 Mbits','Q = 200 Mbits','latex');
xlabel("$l_{\rm{all}}^{\rm{tr}}$ (s)","Interpreter","latex")
ylabel('$f(l_{\rm{all}}^{\rm{tr}})$',"Interpreter","latex")
set(gca,'FontSize',16,'Fontname','Times New Roman')



