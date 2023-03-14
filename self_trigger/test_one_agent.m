% this script test the basic function of the Agent class
clear all; close all; clc;import casadi.*
%% configuration
dyn = @(x,u) [0 0;1 0]*x+[1;0]*u;   % system dynamics
rho = 0.75; mu = 0.5;                % budget dynamics
dim_x = 2; dim_u =1;
% constraints
x_max = 1.1; x_min = -1.1;
u_max = 2; u_min = -2;
r_max = 1; r_min = 0;
delta_max = 1; delta_min = 0.1;
t_global = [0.1:0.1:1];
Q = diag([0,10]); R= 0.1;

% initialization
agent = Agent('dyn',dyn,'rho',rho,'mu',mu,'dim_x',dim_x,'dim_u',dim_u,...
              'x_max',x_max,'x_min',x_min,'u_max',u_max,'u_min',u_min,...
              'r_max',r_max,'r_min',r_min,'delta_max',delta_max,'delta_min',delta_min,...
              'Q',Q,'R',R,'t_global',t_global);
          
%% simulation
sim.x = zeros(dim_x,1); sim.t = 0;
sim.r = 0.5;sim.u = [];
sim.ref = [0;1];
x_init = zeros(size(agent.x));u_init = zeros(size(agent.u));
t_init = [1:agent.horizon]*delta_min; r_init = r_max*ones(1,agent.horizon+1);
dada = [0;0];





while sim.t(end) <=15
    fprintf('time: %.2f\n',sim.t(end));
    state0.x0 = sim.x(:,end); state0.r0 = sim.r(end);
    sol = agent.local_prob(false,state0,x_init,u_init,t_init,r_init,sim.ref(:,end),0);
    
    % forward one step
    u = sol.value(agent.u(:,1)); t = sol.value(agent.t_trigger(:,1));
    r = min(sim.r(end)+t*rho-mu,r_max);
    [~,y] = ode45(@(t,x) dyn(x,u),[0,t],sim.x(:,end));
    
    % logging
    sim.u(:,end+1) = u; sim.t(end+1) = sim.t(end)+t;
    sim.x(:,end+1) = y(end,:)';sim.r(end+1) = r;
    sim.ref(:,end+1) = [0;sign(sign(sin(sim.t(end)/2))+0.1)];
    
    % warm start
    x_init = sol.value(agent.x); u_init = sol.value(agent.u);
    t_init = sol.value(agent.t_trigger); r_init = sol.value(agent.r);
    
end
%% plot
figure(1);clf;
subplot(2,2,1); hold on
plot(sim.t,sim.x(1,:));stairs(sim.t,sim.ref(1,:),'--');
plot(sim.t,[x_max*ones(1,length(sim.t));x_min*ones(1,length(sim.t))],'r')
legend('x_1','reference','bounds');
subplot(2,2,2); hold on
plot(sim.t,sim.x(2,:));stairs(sim.t,sim.ref(2,:),'--');
plot(sim.t,[x_max*ones(1,length(sim.t));x_min*ones(1,length(sim.t))],'r')
legend('x_2','reference','bounds')
subplot(2,2,3); hold on
stairs(sim.t(1:end-1),sim.u);plot(sim.t,[u_max*ones(1,length(sim.t));u_min*ones(1,length(sim.t))],'r')
legend('control inputs','bounds')
subplot(2,2,4);hold on
stairs(sim.t,sim.r);legend('resourse')

%% test polynomial evaluation
x_init = zeros(size(agent.x));u_init = zeros(1,agent.horizon);
t_init = [1:agent.horizon]*delta_min; r_init = r_max*ones(1,agent.horizon+1);
sim.ref = 1; state0.x0 = zeros(dim_x,1); state0.r0 = 0.5;
sol = agent.local_prob(false,state0,x_init,u_init,t_init,r_init,[0;1],0);
x = sol.value(agent.x); t = [0];
tau = [casadi.collocation_points(agent.order,'legendre'),1];
t_poly = [];x_poly = [];
for i = 1:agent.horizon
    temp = sol.value(agent.t_trigger(i));
    t = [t,(temp-t(end))*tau+t(end)];
    t_poly = [t_poly,linspace(t(i),t(i+1)-1e-3,5)];
end
for i = 1: length(t_poly) 
    x_poly = [x_poly,sol.value(agent.poly_eval(agent.t_trigger,t_poly(i)))];
end
%%
figure(3);clf;
subplot(2,1,1);hold on;
plot(t,x(1,:));plot(t_poly,x_poly(1,:));
legend('collocation','polynomial evaluation');
subplot(2,1,2);hold on;
plot(t,x(2,:));plot(t_poly,x_poly(2,:));
legend('collocation','polynomial evaluation');
    