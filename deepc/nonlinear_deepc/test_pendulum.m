% This code is used to test the proposed method in pendulum
clear all; import casadi.*
%% configuration
C = [1,0]; % Output matrix: y = C*x, only first state observed
outputs_dim = size(C,1); % Number of outputs
% Continuous time dynamics
f_u = @(x,u) [x(2);
              -5*sin(x(1))-0.1*x(1)^2+5*abs(cos(x(1)))*u ]; 

% Discretization
Ts_control = 0.04;   
% we use smaller sampling time to improve discretization accuracy
up_samples = 4;
Ts = Ts_control/up_samples;  % discretization sampling frequenct(to improve discretization accuracy)  
%Runge-Kutta 4
k1 = @(x,u) f_u(x,u);
k2 = @(x,u) f_u(x + k1(x,u)*Ts/2,u);
k3 = @(x,u) f_u(x + k2(x,u)*Ts/2,u);
k4 = @(x,u) f_u(x + k3(x,u)*Ts,u);
dyn = @(x,u) x+(Ts/6)*(k1(x,u)+2*k2(x,u)+2*k3(x,u)+k4(x,u));

u = 1*rand(1,4000)-0.5;
x = 2*rand(2,1)-1;    

% down sample to get a trajectory with sampling time Ts_control
for i = 1:length(u)
    % down-sample
    temp = x(:,end);
    for j = 1:up_samples-1
        temp = dyn(temp,u(i));
    end
    x(:,end+1) = dyn(temp,u(i));
end
y = C*x;
clear x; % save memory

% construct Hankel matrix
order = 70;             % length of the control input(state is 1 longer than this) 70
order_meas = 10;        % length of measured control inputs(outputs is order_meas+1)  40
order_pred = order - order_meas;   % length of the predicted part of state
% as adjacent column of Hankel matrix is similar, we down sample the Hankel
% matrix to make the Hankel matrix more informative
hankel_down_samples = 5;    % take column whose index is 1+n*hankel_down_samples

H = build_hankle_struc(u,y,order,order_meas);
H = H(1:hankel_down_samples:end);

% split into training and testing
num_train = 400;     % number of training data
num_test = length(H)-num_train;   % number of testing data
H_train = H(1:num_train);
H_test = H(num_train+1:end);

figure(1);clf
plot(y')
%% data enabled predcition
% The old implementation is different from the framework, whose kernel
% function establishes functions among input and output from different time

% define kernel function acting on one time step
kern_lin = @(x,y) sum(sum(x.*y));         % linear kernel
kern_poly_3 = @(x,y) sum((1+sum(x.*y,1)).^3);   % polynomial kernel
kern_poly_2 = @(x,y) sum((1+sum(x.*y,1)).^2);   % polynomial kernel
kern_exp = @(x,y) sum(exp(sum(x.*y,1)));    % exponential kernl
kern_rbf = @(x,y) sum(exp(-(sum((x-y).^2,1))/6));  % rbf kernel

% define kernel function
kern_u = @(x,y) 0.*kern_lin(x,y)+0.*kern_poly_3(x,y)+0.0*kern_poly_2(x,y)...
                +0.2*kern_rbf(x,y)+1*kern_exp(x,y)+0.01*kern_rbf(x,y)*kern_exp(x,y);      % kernel on inputs
kern_y = @(x,y) 0*kern_lin(x,y)+0.*kern_poly_3(x,y)+1*kern_poly_2(x,y)...
                +0.2*kern_rbf(x,y)+1.*kern_exp(x,y)+0.01*kern_rbf(x,y)*kern_exp(x,y);      % kernel on outputs

gram_meas = zeros(num_train,num_train);
gram_pred = zeros(num_train,num_train);
for i = 1:num_train
    for j = i:num_train
        gram_meas(i,j) = kern_u(H_train(i).u_meas,H_train(j).u_meas)...
                        +kern_y(H_train(i).y_meas,H_train(j).y_meas);
        gram_pred(i,j) = kern_u(H_train(i).u_pred,H_train(j).u_pred)...
                        +kern_y(H_train(i).y_pred,H_train(j).y_pred);
        % symmetric
        gram_meas(j,i) = gram_meas(i,j);
        gram_pred(j,i) = gram_pred(i,j);
    end
end

%% ======== define optimization problem =========

% ------------ problem to do prediction -----------
opti_pred = casadi.Opti();
g = opti_pred.variable(num_train,1); % the weight 
% 
pred = struct; 
% kernel evaluation of the cross the traj and the data in the measured part
pred.kern_meas = opti_pred.parameter(num_train,1);    
pred.norm_meas = opti_pred.parameter(1);    % RKHS norm contributed by the measured part
pred.y_pred = opti_pred.variable(size(H_train(i).y_pred,1),size(H_train(i).y_pred,2));

% reciprocal norm square of the sequence in RKHS
% pred.a = opti_pred.variable(1);         

% calculate some term of 
pred.norm_pred = kern_y(pred.y_pred,pred.y_pred);
pred.kern_pred = [];
for i = 1:num_train
    pred.kern_pred = [pred.kern_pred;kern_y(H_train(i).y_pred,pred.y_pred)];
end

% consider the quatient space
% opti_pred.subject_to(g'*(gram_meas+gram_pred)*g <= (pred.norm_meas+pred.norm_pred));
% scaling of the prediction whole trajectory
% opti_pred.subject_to(1==pred.a*(pred.norm_meas+pred.norm_pred));
% opti_pred.subject_to(pred.a>=0); 
% function
loss = g'*(gram_meas+gram_pred)*g ...
        -2*(pred.kern_pred+pred.kern_meas)'*g...
        +(pred.norm_meas+pred.norm_pred);
opti_pred.minimize(loss);

% setup solver
ops = struct;
ops.ipopt.print_level = 0;
ops.ipopt.tol = 1e-3;
opti_pred.solver('ipopt',ops);

%% solving optimization problem
logs = [];
for i = 1:45:num_test
    logs(end+1).index = i;
    if length(logs)>9
        break;
    end
    % recover the weight
    cross_meas_temp = zeros(num_train,1);
    for j = 1:num_train
        cross_meas_temp(j) = kern_u(H_train(j).u_meas,H_test(i).u_meas)...
                            +kern_u(H_train(j).u_pred,H_test(i).u_pred)...
                            +kern_y(H_train(j).y_meas,H_test(i).y_meas);
    end
    auto_meas_temp = kern_u(H_test(i).u_meas,H_test(i).u_meas) ...
                    +kern_u(H_test(i).u_pred,H_test(i).u_pred) ...
                    +kern_y(H_test(i).y_meas,H_test(i).y_meas);
    opti_pred.set_value(pred.kern_meas,cross_meas_temp);
    opti_pred.set_value(pred.norm_meas,auto_meas_temp);
    
    opti_pred.set_initial(pred.y_pred,repmat(H_test(i).y_meas(:,end),1,order_pred));

    sol = opti_pred.solve();
    logs(end).y_pred = sol.value(pred.y_pred);
    
end
%% plotting
figure(3);clf;
for i = 1:length(logs)
    if i>9
        break
    end
    temp = logs(i).index;
    temp_pred = logs(i).y_pred;
    subaxis(3,3,i, 'Spacing', 0.03, 'Padding', 0.0, 'Margin', 0.025);hold on
    plot((order_meas+[0:size(temp_pred,2)])*Ts,[H_test(temp).y_meas(:,end),temp_pred],'LineWidth',2);
    plot((order_meas+[0:size(temp_pred,2)])*Ts,[H_test(temp).y_meas(:,end),H_test(temp).y_pred],'--','LineWidth',2);
    plot(([0:order_meas])*Ts,H_test(temp).y_meas,'k-','LineWidth',2);
    legend('predicted','real','measured part','Location','southeast');
    axis([0 (order+1)*Ts -0.3 1])
    xlabel('time (s)');ylabel('y')
    axis tight
end
%% save for tikz plotnan
% ind = [2,3,4,8];
% dada = table;
% dada.t = [0:order]'*Ts;
% temp = ind(1);
% dada.meas1 = [H_test(logs(temp).index).y_meas';nan(order_pred,1)];
% dada.pred1 = [nan(order_meas,1);H_test(logs(temp).index).y_meas(end);logs(temp).y_pred'];
% dada.real1 = [nan(order_meas,1);H_test(logs(temp).index).y_meas(end);H_test(logs(temp).index).y_pred'];
% 
% temp = ind(2);
% dada.meas2 = [H_test(logs(temp).index).y_meas';nan(order_pred,1)];
% dada.pred2 = [nan(order_meas,1);H_test(logs(temp).index).y_meas(end);logs(temp).y_pred'];
% dada.real2 = [nan(order_meas,1);H_test(logs(temp).index).y_meas(end);H_test(logs(temp).index).y_pred'];
% 
% temp = ind(3);
% dada.meas3 = [H_test(logs(temp).index).y_meas';nan(order_pred,1)];
% dada.pred3 = [nan(order_meas,1);H_test(logs(temp).index).y_meas(end);logs(temp).y_pred'];
% dada.real3 = [nan(order_meas,1);H_test(logs(temp).index).y_meas(end);H_test(logs(temp).index).y_pred'];
% 
% temp = ind(4);
% dada.meas4 = [H_test(logs(temp).index).y_meas';nan(order_pred,1)];
% dada.pred4 = [nan(order_meas,1);H_test(logs(temp).index).y_meas(end);logs(temp).y_pred'];
% dada.real4 = [nan(order_meas,1);H_test(logs(temp).index).y_meas(end);H_test(logs(temp).index).y_pred'];
% 
% writetable(dada, 'data/pendulum_pred.dat', 'Delimiter','space');
% %%
% dada = table;
% dada.t = Ts_control*vec([1:500]);
% dada.y = y(3001:3500)';
% writetable(dada, 'data/pendulum_data.dat', 'Delimiter','space');