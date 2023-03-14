% This is the code to run robust mpc with hand derived robust formulation
clear all; close all; clc;
load('motor_model.mat');
%% recovere function handel
f_u = @dyn_motor_scaled; % Dynamics, defined at the end of the script

% Discretize
Ts = 0.01;
%Runge-Kutta 4
k1 = @(t,x,u) (  f_u(t,x,u) );
k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*Ts/2,u) );
k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*Ts/2,u) );
k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*Ts,u) );
f_ud = @(t,x,u) ( x + (Ts/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );
%% define the optimization problem
% ---- formulate the condensed form 
mpc_horizon = 10;
mpc_obs = Or(2:(mpc_horizon+1)*outputs_dim,:); % extended observability matrix
mpc_ctrl = [];
for i = 1:mpc_horizon
    mpc_ctrl = [mpc_ctrl;H_fi(:,end-u_dim*(i+1):end-u_dim-1),zeros(outputs_dim,(mpc_horizon-i)*u_dim)];
end
mpc_x0 = sdpvar(order,1); % initial states
mpc_v = sdpvar(u_dim*mpc_horizon,1); % nominal inputs
mpc_l = sdpvar(u_dim*mpc_horizon,outputs_dim*mpc_horizon,'full').*...
      (kron(tril(ones(mpc_horizon)-eye(mpc_horizon)),ones(u_dim,outputs_dim)));
mpc_w = sdpvar(order,1); % uncertainty
mpc_y_norm = mpc_obs*mpc_x0+mpc_ctrl*mpc_v;
mpc_y_obs = mpc_y_norm+(mpc_obs+mpc_l*mpc_obs)*mpc_w;
mpc_u = mpc_v+mpc_l*mpc_obs*mpc_w; % actual control inputs
mpc_y_ss = sdpvar(1,1); % set point
% constraints
cons = [-1<=mpc_y_obs<=1,-1<=mpc_u<=1];
% loss
objective = (mpc_y_norm-mpc_y_ss)'*(mpc_y_norm-mpc_y_ss)+mpc_v'*mpc_v*1e-5;
% configuration
ops = sdpsettings('solver','gurobi','verbose',0);


%% close loop simulation
rng(11234)
% generate the initial data
x0_temp = 2*rand(x_dim,1)-1;
y_sim = C_hat*x0_temp;
u_sim = rand(u_dim,n_delay)*2-1;
for i = 1:n_delay
    x0_temp = f_ud(0,x0_temp,u_sim(:,i));
    y_sim = [C_hat*x0_temp;u_sim(:,i);y_sim];
end
x_log = [x0_temp];
u_log = [];
y_ss_log = [];
for i = 1:150
    i
    x_lift = [];
    w_bound = [];
    for j = 1:order
        [temp,temp_sd,~] = predict(hyp_trained{j},y_sim','Alpha',0.01);
        x_lift = [x_lift;temp];
        w_bound = [w_bound;temp_sd];
    end
    % update the bound
    cons_w = [-w_bound<=mpc_w<=w_bound];
    % update the robust set
    [cons_robust,h] = robustify([cons,cons_w],objective,[],mpc_w);
    % solve optimization problem
    temp_y_ss = -0.15+0.55*sign(sin(0.015*i*2*pi));
    y_ss_log = [y_ss_log,temp_y_ss];
    optimize([cons_robust,mpc_x0==x_lift,mpc_y_ss==temp_y_ss],h,ops);
    
    % apply control inputs
    x_temp = f_ud(0,x_log(:,end),value(mpc_v(1:u_dim)));
    x_log = [x_log,x_temp];
    u_log = [u_log,value(mpc_v(1:u_dim))];
    
    % update the log of previous inputs
    y_sim = [C_hat*x_temp;u_log(:,end);y_sim(1:end-u_dim-outputs_dim)];
end
%%
figure(1);clf;hold on;
plot([0:size(x_log,2)-2]*Ts,x_log(2,1:end-1));
plot([0:size(x_log,2)-2]*Ts,y_ss_log','--');
xlabel('time');ylabel('outputs')

figure(2);clf;hold on;
plot([0:size(x_log,2)-2]*Ts,u_log);
plot([0:size(x_log,2)-2]*Ts,ones(length(u_log),1),'r--')
plot([0:size(x_log,2)-2]*Ts,-ones(length(u_log),1),'r--')


ddd = table;
ddd.t = [0:size(x_log,2)-2]'*Ts;
ddd.x = x_log(2,1:end-1)';
ddd.u = u_log';
ddd.uL = -ones(length(u_log),1);
ddd.uU = ones(length(u_log),1);
ddd.xL = -ones(length(u_log),1);
ddd.xU = ones(length(u_log),1);
ddd.ref = y_ss_log';
% writetable(ddd, '../docs/tikz/motor_mpc.dat', 'Delimiter','space');



%% motor dynamics
% x1 - current in [-1,1]
% x2 - angular velocity in [-1,1]
% u  - control input in [-1,1]

function f = dyn_motor_scaled( t,x,u )
f = [   19.10828025-39.3153*x(1,:)-32.2293*x(2,:).*u;
       -3.333333333-1.6599*x(2,:)+22.9478*x(1,:).*u ];
end

%Model from ([1], Eq. (9))
%   dx1 = -(Ra/La)*x(1) - (km/La)*x(2)*u + ua/La
%   dx2   = -(B/J)*x(2) + (km/J)*x(1)*u - taul/J;
%   y = x1
% Parameters La = 0.314; Ra = 12.345; km = 0.253; J = 0.00441; B = 0.00732; taul = 1.47; ua = 60;
% Constraints (scaled to [-1,1] in the final model)
% x1min = -10; x1max = 10;
% x2min = -100; x2max = 100;
% umin = -4; umax = 4;
% [1] S. Daniel-Berhe and H. Unbehauen. Experimental physical parameter
% estimation of a thyristor driven DC-motor using the HMF-method.
% Control Engineering Practice, 6:615?626, 1998.
    
