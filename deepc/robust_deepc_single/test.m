% this is the first trial of robust closed-loop deepc with measured process
% noise in a 2nd order system, we don't consider measurement noise in this
% test, but should be possible to handle it in the existed DeePC framework
% The model considered is
%           x+ = Ax +Bu+ Ew
%           y = Cx 
clear all; close all; clc
%% configuration of the problem
% we first define the linear system in continuous time 
tau_s = 0.3;  ... time constant
eta =  0.8; ... damping ratio
Kp = 1;     ... input gain
% continuous time state space model
A = [0 1; -1/tau_s^2 -2*eta/tau_s]; B = [0;Kp/tau_s^2];
C = [1 0]; 
Ts = 0.1;  ... sampling time

sys = ss(A,B,C,[]);
sys = c2d(sys,Ts); ... discretization zoh
E = sys.B;

dim_y = size(C,1); dim_u = size(B,2); dim_w = size(E,2);

% configuration of control and process noise 
w_min = -0.5; w_max = 0.5;  ... bound for noise
u_min = -5; u_max = 5;      ... bound for the control input
y_min = -0.5; y_max = 0.5;      ... bound for the outputs
Qy = 10; R = 1e-3;              ... stage cost

% ------ configuration of the Hankel matrix ------
num_data = 80;     ... number of data
depth_init = 3;      ... number of inputs use for initialization, 
horizon = 10;       ... prediction horizon for MPC
depth = horizon+depth_init;   ... depth of the Hankel matrix w.r.t u, depth for y is depth+1
% generate data
u = (u_max-u_min)*rand(dim_u,num_data)+u_min;
w = (w_max-w_min)*rand(dim_u,num_data)+w_min;
temp = [zeros(2,1)];    ... temp for state trajectory
for i = 1:num_data
    temp(:,end+1) = sys.A*temp(:,end)+sys.B*u(:,i)+E*w(:,i);
end
y = sys.C*temp;
figure(1);clf;
subplot(2,1,1); plot(y);
title('training data');legend('output')
subplot(2,1,2); hold on;
plot(u);plot(w);legend('input','disturbance')
clear temp; ...save memory
[Hu,Hw,Hy,Hbar] = build_hankel(u,w,y,depth);

% some matrix used for the MPC formulation
Q_tilde = kron(eye(horizon),Qy); R_tilde = kron(eye(horizon),R);
% inequality constraints for outputs and inputs F_y*y<=f_y, F_u*u<=f_u
F_u = kron(eye(horizon),[eye(dim_u);-eye(dim_u)]);
f_u = kron(ones(horizon,1),[u_max*ones(dim_u,1);-u_min*ones(dim_u,1)]);
F_y = kron(eye(horizon),[eye(dim_y);-eye(dim_y)]); 
f_y = kron(ones(horizon,1),[y_max*ones(dim_y,1);-y_min*ones(dim_y,1)]);
% set of disturbance
F_w = kron(eye(horizon),[eye(dim_w);-eye(dim_w)]);
f_w = kron(ones(horizon,1),[w_max*ones(dim_w,1);-w_min*ones(dim_w,1)]);
%% ------ implementation of the closed-loop feedback DeePC
% construction of the null space
[Q,~] = qr([Hw(1:dim_w*depth_init,:);Hbar]');
% only take the related components
Q = Q(:,dim_w*depth_init+(dim_u+dim_y)*(depth_init+1)+1:size(Hbar,1)+dim_w*depth_init);

% ====== define the optimization problem ======
% tracking reference
ref = sdpvar(dim_y,1,'full');
% parametrization of the feedback matrix(lower block triangular)
temp = kron(tril(ones(horizon)),ones(dim_u+dim_y,1));
temp = temp(1:end-dim_u,:); ... the last noise wn can only change the last y_{n+1}
%         ones(size(Hbar,2)-size(Hbar,1),horizon)];
Kbar = sdpvar(size(Q,2),horizon,'full').*temp;
% the causal feedback control law
K = Q*Kbar;
% nomial control input 
g = sdpvar(size(Hbar,2),1,'full');
% initial states 
u0 = sdpvar(dim_u*depth_init,1,'full'); y0 = sdpvar(dim_y*(depth_init+1),1,'full');
w0 = sdpvar(dim_w*depth_init,1,'full');
% nominal trajetory in prediction
u = Hu*g; u_pred_norm = u(dim_u*depth_init+1:end,1);
y = Hy*g; y_pred_norm = y(dim_y*(depth_init+1)+1:end,1);
w_init = Hw(1:dim_w*depth_init,:)*g; 
% ====== Constraints ======
cons = [];
% initial states
cons = [cons;u(1:dim_u*depth_init)==u0];
cons = [cons;y(1:dim_y*(depth_init+1))==y0];
cons = [cons;w_init==w0]; 
% constraints for the nominal trajectory(noise-free)
cons = [cons;Hw(dim_w*depth_init+1:end,:)*g==zeros(horizon*dim_w,1)];
% constraints for the feedback law parametrization
cons = [cons;Hw(dim_w*depth_init+1:end,:)*K==eye(horizon*dim_w)];
% robust constraint (F_u*u_pred<=f_u, F_y*y_pred<=f_y), where u_pred and
% y_pred have both nominal part and the feedback part

% ------ dual variables ------
lambda_y = sdpvar(size(F_w,1),size(F_y,1),'full'); ... each column corresponds to one robust constraint
lambda_u = sdpvar(size(F_w,1),size(F_u,1),'full'); ... each column correspongs to one robust constraint   
% --------- robust constraint -----------
% w.r.t y
cons = [cons;lambda_y'*f_w<=f_y-F_y*y_pred_norm];
for i = 1:size(F_y,1)
    % we can get rid of the loop, we keep to loop to make the code more readable
    cons = [cons;F_w'*lambda_y(:,i)==(F_y(i,:)*Hy(dim_y*(depth_init+1)+1:end,:)*K)'];
    cons = [cons; lambda_y(:,i)>=zeros(size(F_w,1),1)];
end
% w.r.t u
cons = [cons;lambda_u'*f_w<=f_u-F_u*u_pred_norm];
for i = 1:size(F_u,1)
    cons = [cons;F_w'*lambda_u(:,i)==(F_u(i,:)*Hu(dim_u*depth_init+1:end,:)*K)'];
    cons = [cons; lambda_u(:,i)>=zeros(size(F_w,1),1)];
end
ref_tilde = kron(ones(horizon,1),ref);
obj = (y_pred_norm-ref_tilde)'*Q_tilde*(y_pred_norm-ref_tilde)...
      + u_pred_norm'*kron(eye(horizon),R)*u_pred_norm;

opts = sdpsettings('solver','gurobi','verbose',1);
ctrl = optimizer(cons,obj,opts,{u0,y0,w0,ref},{u_pred_norm,y_pred_norm,K});

%% closed loop simulation
nsteps = 46;
sim.ref = 0.5;
sim.x = zeros(2,1);
sim.u = [];
sim.w = [];
% generate the initialization part for DeePC
for i = 1:depth_init
    sim.u(:,end+1) = 0.01*((u_max-u_min)*rand(dim_u,1)+u_min);
    sim.w(:,end+1) = 0.1*((w_max-w_min)*rand(dim_w,1)+w_min);
    sim.x(:,end+1) = sys.A*sim.x(:,end)+sys.B*sim.u(:,end)+E*sim.w(:,end);
end

for i = 1:nsteps
    % get the information for the controller
    temp_u_init = vec(sim.u(:,end-depth_init+1:end));
    temp_y_init = vec(sys.C*sim.x(:,end-depth_init:end));
    temp_w_init = vec(sim.w(:,end-depth_init+1:end));
    
    [u_opt,flag] = ctrl({temp_u_init,temp_y_init,temp_w_init,sim.ref(:,end)});
    if flag ~= 0
        msg = yalmiperror(flag);
        error(msg);
    end
    
    % forward one-step
    sim.w(:,end+1) = (w_max-w_min)*rand(dim_w,1)+w_min;
    sim.u(:,end+1) = vec(u_opt{1}(1:dim_u));
    sim.x(:,end+1) = sys.A*sim.x(:,end)+sys.B*sim.u(:,end)+E*sim.w(:,end);
    sim.ref(:,end+1) = 0.5*sign(sin(0.2*i));
    
end
%% standard robust mpc(refer to code on Yalmip tutorial)
u_rb_mean = sdpvar(horizon,1);
K_rb = sdpvar(horizon,horizon,'full').*(tril(ones(horizon))-eye(horizon));
w_rb = sdpvar(horizon,1);
x0_rb = sdpvar(2,1);
ref_rb = sdpvar(1);
u_rb = K_rb*w_rb + u_rb_mean;

y_rb = [];
x_rb_mean = x0_rb;
xk = x0_rb;
obj = 0;
for k = 1:horizon
    x_rb_mean = [x_rb_mean,sys.A*x_rb_mean(:,end)+sys.B*u_rb_mean(k)];
    xk = sys.A*xk + sys.B*u_rb(k)+E*w_rb(k);
    y_rb = [y_rb;sys.C*xk];
    obj = (sys.C*x_rb_mean(:,end)-ref_rb)*Qy*(sys.C*x_rb_mean(:,end)-ref_rb)...
            +u_rb_mean(k)*R*u_rb_mean(k)+obj;
end
F = [y_min<=y_rb <= y_max, u_min <= u_rb <= u_max];
G = [w_min <= w_rb <= w_max];

opts = sdpsettings('solver','gurobi','verbose',1);   

ctrl_rb = optimizer([F,G, uncertain(w_rb)],obj,opts,{x0_rb,ref_rb},u_rb(1));
%% closed loop simulation
xk_rb = sim.x(:,depth_init+1);
for i = 1:nsteps
    xk_rb = [xk_rb sys.A*xk_rb(:,end) + sys.B*ctrl_rb{xk_rb(:,end),sim.ref(i)} ...
            + E*sim.w(i+depth_init)];
end
%% plot
figure(2);clf;hold on
set(gca,'fontsize',18)
plot([1:(size(sim.x,2)-depth_init)],sys.C*sim.x(:,depth_init+1:end),'b','LineWidth',1.5);
plot(1:size(xk_rb,2),sys.C*xk_rb,'g--','LineWidth',3);
plot(2:(size(sim.ref,2)+1),sim.ref,'k','LineWidth',1);
plot(1:size(sim.ref,2),y_min*ones(1,size(sim.ref,2)),'r--','LineWidth',1)
plot(1:size(sim.ref,2),y_max*ones(1,size(sim.ref,2)),'r--','LineWidth',1)
legend('output robust DeePC','standard robust MPC','reference','constraints')
xlim([1,nsteps])
%% save data for tikz plotting
ddd =table;
ddd.t = vec([1:(size(sim.x,2)-depth_init)]);
ddd.deepc = vec(sys.C*sim.x(:,depth_init+1:end));
ddd.mpc = vec(sys.C*xk_rb);
ddd.max = vec(y_max*ones(1,size(sim.ref,2)));
ddd.min = vec(y_min*ones(1,size(sim.ref,2)));
ddd.ref = vec([sim.ref(1),sim.ref(1:end-1)]);
writetable(ddd, 'tikz/2dsys.dat', 'Delimiter','space');




  
  
  
  
  
  