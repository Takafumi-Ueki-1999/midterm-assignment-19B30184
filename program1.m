n = 200;
epoch = 100;
x = 3 * (rand(n,4) - 0.5);
y = (2 * x(:,1) - 1 * x(:,2) + 0.5 + 0.5 * randn(n,1)) > 0;
y = 2 * y -1;
eta = 0.01; %学習率
w = rand(4,1); %重みをランダムで初期設定
w_hat = w;
w_newton = w;
w_newton_hat = w_newton;

X = 1:epoch;
Y = zeros(1,epoch);
Y_newton = zeros(1,epoch);

for i = 1:epoch %batch steepest gradient method
    Y(i) = J(w,x.',y,n);
    w = w - eta*J_dot(w,x.',y,n); %重みを更新
    if J(w_hat,x.',y,n) > J(w,x.',y,n)
        w_hat = w;
    end
    Y_newton(i) = J(w_newton,x.',y,n); %newton method
    d = J_dot_dot(w_newton,x.',y,n)\J_dot(w_newton,x.',y,n);
    w_newton = w_newton - eta*d; %重みを更新
    if J(w_newton_hat,x.',y,n) > J(w_newton,x.',y,n)
        w_newton_hat = w_newton;
    end
end
Y = abs(Y - J(w_hat,x.',y,n));
Y_newton = abs(Y_newton - J(w_newton_hat,x.',y,n));

semilogx(X,Y); %結果を出力
hold on;
semilogx(X,Y_newton);
xlabel('t','Interpreter','latex');
ylabel("$|J(w^{(t)}) - J(\hat{w})|$",'Interpreter','latex');
legend('batch steepest gradient method','newton method');

function J = J(w,x,y,n)
    temp = 0;
    for i =1:n
        temp = log(1+exp(-y(i)*(w.')*x(:,i))) + temp;
    end
    J = temp + (w.')*w; 
end

function J_dot = J_dot(w,x,y,n)
    temp = 0;
    for i =1:n
        temp = (exp(-y(i)*(w.')*x(:,i))*(-y(i)*x(:,i)))/(1 + exp(-y(i)*(w.')*x(:,i))) + temp;
    end
    J_dot = temp + 2*w; 
end
    
function J_dot_dot = J_dot_dot(w,x,y,n)
    temp = 0;
    for i =1:n
        temp = (exp(-y(i)*(w.')*x(:,i)))*x(:,i)*(x(:,i).')/(1 + exp(-y(i)*(w.')*x(:,i))) + temp;
    end
    J_dot_dot = (1/n)*temp + eye(4); 
end
