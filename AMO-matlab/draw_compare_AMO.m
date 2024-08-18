figure(1)
X = categorical({'50','100','150','200','250'});
X = reordercats(X,{'50','100','150','200','250'});
Y1 = [2 4 6 8 10;
      2 4 6 8 10;
      5 10 16 20 20];
bar(X,Y1)
xlabel('$Q$ (Mbits)',"Interpreter","latex")
ylabel('$N$',"Interpreter","latex")
legend('AMO','LINGO','fmincon')
set(gca,'FontSize',16,'Fontname','Times New Roman')


figure(2)
X = categorical({'50','100','150','200','250'});
X = reordercats(X,{'50','100','150','200','250'});
Y2 = [0.001456 0.001768 0.001754 0.001812 0.001887;
      2.21 0.20 0.70 0.29 0.33;
      0.097540 0.104819 0.102641 0.047505 0.052709];
bar(X,Y2)
xlabel('$Q$ (Mbits)',"Interpreter","latex")
ylabel('Computational time (s)')
legend('AMO','LINGO','fmincon') 
set(gca,'FontSize',16,'Fontname','Times New Roman')
