% this script is used to test the idea of shifting scheduling of a load
% pattern. 
% In this code, we use kernel based smooth spling to achieve both
% integration and interpolation. The reason was that the my previous trial
% with piece-wise polynomial leads to really bad regularity
%
% This code validates both the concept of piece-wise kernel interpolation
% and global grid evaluation 

% THIS CODE HAS AN MINOR ERROR AS TS CAN BE SMALLER THAN T0
clear; close all; clc; import casadi.*;
%% basic configuration 
% load curtailment tube
tube_width = 0.5;
% nominal pattern of the tube
traj = @(t) 2*sign(t+1e-6) - sign(t-1+1e-6) - sign(t-2+1e-6); ... sharpter shape with 1e-6
delay_max = 3;

% ------ components for the optimization problem ------
% ---- delay cost ----
loss_delay = @(t) 1*max(t-0.3,0)^2;
% ---- curtailment cost ----
loss_curt = @(delta) 1*delta;
% ---- electricity price ---
loss_pow = @(t,x) max(2*exp(-(t-2))-1,0)*x;

% number of global time grid points
num_grid = 25;

% ====== Other setup for integration and interpolation======
delta_local = 1;
root = linspace(0,delta_local-1e-3,8);
num_root = length(root);
kern = @(x,y) exp(-15.5*(x-y).^2);
Gram = [];
for i = 1:length(root)
    Gram(end+1,:) = kern(root(i),root);
end

% linear map from evalution to weight space
a_map = inv(Gram);

% ---- integration operator ----
% validation is coeff_quad*a_map*ones(num_root,1)\approx delta_local
coeff_quad = [];
for i = 1:length(root)
    temp = @(t) kern(root(i),t);
    coeff_quad(end+1) = integral(temp,0,delta_local-1e-3);
end

% validation of the interpolant
% y = 5*rand(num_root,1);
% temp = [0:0.05:1];
% temp_u = [];
% for i = 1:length(temp)
%     temp_u(end+1) = kern(temp(i),root)*a_map*y;
% end
% figure(1);clf;hold on;
% plot(root,y);
% plot(temp,temp_u);

%% configuration of the optimization problem
% REMARK ON THIS IMPLEMENTATION
% Grid 1: absolute time grid. This grid is used to coordinate all the agents,
% whose initial point is the current time.
% Grid 2: absolute time grid. This grid is used to define the delay with
% only two points, the time of request and the begin time of execution
% Grid 3: relative time grid. This grid is used to optimize the load, and
% the initial point is 0 w.r.t the begin time of execution.
% The benefit of this gridding is two-folded. Firstly, the delay upper
% bound is fixed in Grid 2. Secondly, we can easily map the global time
% Grid 1 to the local time Grid 3, which simplifies the evaluation of the
% load profile on the global Grid 1.

opti = casadi.Opti();

% ======Definition of the time grid and the variables/parameter on the
% corresponding grid =======
% ------ Grid 1: global time grid ------
t0 = opti.parameter(1);
delta_global = opti.parameter(1);
grid_global = t0 + [0:num_grid]*delta_global;

% ------ Grid 2: absolute local grid ------
tr = opti.parameter(1); ... request time
ts = opti.variable(1);  ... starting time of execution

% ------ Grid 3: relative local grid ------
grid_local = kron(0:1,ones(1,num_root))+kron(ones(1,2),root);
grid_local = [grid_local,grid_local(end-num_root+1)+delta_local];

u = opti.variable(1,2*num_root);   ... load profile

% ====== cons and loss =======
% ------ on Grid 1 ------
% ====== REMARK ======
% WE CAN CALCULATE THE POWER COST IN THE LOCAL GRID, WE USE THE GLOBAL GRID
% HERE TO TEST THE FUNCTIONALITY OF MULTIPLE AGENT COORDINATION
trap_quad = [ones(1,num_grid),0]+[0,ones(1,num_grid)]/2;  ... Trapezoidal quadrature for integration
temp = [];
for i = 1:length(grid_global)
    temp = [temp,...
        loss_pow(grid_global(i),pcw_poly(grid_global(i)-ts,grid_local,u,num_root,a_map,root,kern))];
end
loss = temp*trap_quad'*delta_global;

% ------ on Grid 2 ------
opti.subject_to(delay_max>=ts-tr>=0);
loss = loss + loss_delay(ts-tr);

% ------ on Grid 3 ------
for n_iter = 1:2
    temp_cost = [];
    temp_ind = (n_iter-1)*num_root;
    for i = 1:num_root
        % get the desired profile
        temp = traj(grid_local(temp_ind+i));
        % input constraint
        opti.subject_to((temp-tube_width)<=u(temp_ind+i)<=temp);
        temp_cost = [temp_cost;loss_curt(temp-u(temp_ind+i))];
    end
    loss = loss +coeff_quad*a_map*temp_cost;
end

opti.minimize(loss);
% setup the solver
opts = struct;
opts.ipopt.print_level = 0;
opts.print_time = false;
opts.ipopt.max_iter = 1e4;
opts.ipopt.tol = 1e-8;
opti.solver('ipopt', opts);
%% closed loop simulation
sim = struct;
sim.t = 0;
sim.tr = 0;
sim.ts = 3;
sim.u = [];
len_u = length(u);
ind_fix = 0;    ... the last interval that has been executed
logs = [];
flag_start = true;  ... use to fix ts when it just start

opti.set_initial(ts,delay_max);
opti.set_initial(u,traj(grid_local(1:end-1)))
while sim.t(end)< 6
    % update boundary condition
    opti.set_value(delta_global,.5);
    opti.set_value(tr,sim.tr);
    opti.set_value(t0,sim.t(end));
    if sim.ts < sim.t(end)
        % ------ load started ------
        if flag_start
            % just started, fixed the ts
            opti.subject_to(ts==sim.ts);
            flag_start = false;
        end
        ind = find(sim.ts+grid_local>=sim.t(end),1); 
        if isempty(ind)
            ind = num_root*2+1;
        end
        ind = floor((ind-1)/num_root);
        if ind>ind_fix
            % execute a new interval
            sim.u = sol.value(u(1:ind*num_root));  ... get all the input happened
            for i = ind_fix*num_root+1:ind*num_root
                % fix all the input used recently
                opti.subject_to(u(i)==sim.u(i));
            end
            ind_fix = ind;
        else 
            % no new interval executed, no need to solve again
            % do nothing
        end
    end
    sol = opti.solve();

    sim.t(end+1) = sim.t(end)+.5;
    sim.ts = sol.value(ts);
    logs(end+1) = sim.ts;

    % warm start
    opti.set_initial(u,sol.value(u));
    opti.set_initial(ts,sol.value(ts));
end

%% plotting
figure(1);clf;hold on;
temp_t = linspace(sim.t(1)+1e-6,sim.t(end),100);

temp = [];
for i = 1:length(temp_t)
    temp(:,end+1) = pcw_poly(temp_t(i)-sim.ts,grid_local,sim.u,num_root,a_map,root,kern);
end
plot(temp_t,traj(temp_t),'r');
plot(temp_t,temp,'b-*')
plot(temp_t,traj(temp_t-sim.ts),'r--')
legend('request profile','load profile','shifted request profile')


%% helper function
function y = pcw_poly(t,grid,u,num_root,a_map,root,kern)
    % evalution of the piece-wise kernel interpolation
    y = 0;
    for i = 1:num_root:length(u)
        temp = kern(t-grid(i),root)*a_map*vec(u(i:i+num_root-1));
        y = y+temp*bump(t,grid(i),grid(i+num_root));
    end
end
