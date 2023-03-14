% this script tries a different way of collocation method, where we don't
% directly look into the differential dynamics of the system, but the
% differential dynamics of the integration
clear; close all; clc;
%% configuration of the problem
% ------ system dynamics ------
% double integrator
A = [0,1; 0 0]; B_u = [0;1];
B_w = B_u; 
C = [1,0];

% noise property
w_max = 0.3; w_min = -0.3; ... 
 
% ====== configuration of the MPC controller =======
% ----- basic setup -----
rho = 0.8; mu = 0.4;                % budget dynamics
dim_x = size(A,1); dim_u =size(B_u,2); dim_w = size(B_w,2);
horizon = 8;           % prediction horizon
% constraints
x_max = [1;10]; x_min = -[1;10];      % state constraints on the second output
u_max = 5; u_min = -5;
r_max = 1.; r_min = 0;
delta_max = 1.5; delta_min = 0.1;
Q_y = 10; Q_u= 0;

% configuration of the perturbation set
R_w = ((w_max-w_min)/2)^2;  ... the PSD matrix of the zero mean ellipsoidal noise
    
% design feedback controller
temp = ss(A,B_u,C,[]);
temp = c2d(temp,mu/rho);

K = -place(temp.A,temp.B,[0.3,0.4]); ... we use A+BK in this code

%% construction of the optimization problem
opti = casadi.Opti();

% ------ configuration of collocation method ------
order = 3;      % collocation order
tau = casadi.collocation_points(order, 'legendre');    % collocation points except 0
% coeff_diff: i,j entry is derivative of i-th basis evaluted at j-th point
% coeff_cont: evaluation at 1;  coeff_quad: integration up to 1
[coeff_diff,coeff_cont,coeff_quad] = casadi.collocation_coeff(tau);

% ======== decision variables ========
% ------- norminal decision variables -------
x = opti.variable(dim_x,horizon*(order+1)+1);   ... center of the uncertainty set
u = opti.variable(dim_u,horizon);               ... nominal control input
delta = opti.variable(1,horizon);               ... triggering time
r = opti.variable(1,horizon+1);                 ... resource
cost = opti.variable(1,horizon+1);              ... cost

% ------ variable related to feedback -------
% NOTE THAT WE THE EVALUATION AT THE COLLOCATION POINT
G = opti.variable(dim_x^2,horizon*order); ... fundamental solution related to noise
% fundamental solution w.r.t feedback (no feedback in the first interval)
G_fb = opti.variable(dim_x^2,(horizon-1)*order);

% variance at the triggered point
P = opti.variable(dim_x^2,horizon);
% some extra constraint to help
for i = 1:horizon
    temp = diag(mat(P(:,i)));
    opti.subject_to(temp>=0);
end


% some terms related to sum of ellipsoids
% weight of the fundamental solution
lambda_f_sqrt = opti.variable(1,order*horizon); 
lambda_cont_sqrt = opti.variable(1,horizon-1); % sum weight at the end of interval
% direct enforcement of positivity
epsilon = 1; ... bias term to avoid ill conditioning (will be a bit sub postimal, but much easier to solve)
lambda_f = ones(1,horizon*order);...lambda_f_sqrt.^2+epsilon;
lambda_cont = 0.5*ones(1,horizon-1);...lambda_cont_sqrt.^2+epsilon;
% upper bound constraints
% opti.subject_to(lambda_cont<=1-epsilon);


% ============ paramters =========
x0 = opti.parameter(dim_x,1);
r0 = opti.parameter(1); 
ref = opti.parameter(1);

% ========== define dynamics ==========
% -------- initial condition --------
opti.subject_to(x(:,1)==x0);
opti.subject_to(r(1)==r0);
opti.subject_to(cost(1)==0);
G_init = vec(eye(dim_x)); ... initial states for all the fundamental solutions (G,G_fb)
for n_time = 1:horizon
    % -------- nominal part --------
    temp_ind = (n_time-1)*(order+1);
    temp_x = x(:,temp_ind+1:temp_ind+order+1); ... center of the ellipsoid in this interval
    dx = A*temp_x(:,2:end)+B_u*repmat(u(:,n_time),1,order);
    opti.subject_to(temp_x*coeff_diff==dx*delta(n_time));
    % continuity at 1
    opti.subject_to(temp_x*coeff_cont==x(:,n_time*(order+1)+1));
    % cost evaluation at collocation points
    temp = [];
    for i = 1:order
        temp = [temp,(C*temp_x(:,i+1)-ref)'*Q_y*(C*temp_x(:,i+1)-ref)+u(n_time)'*Q_u*u(n_time)];
    end
    opti.subject_to(cost(n_time+1)==cost(n_time)+temp*coeff_quad*delta(n_time));
    
    % ------- resource dyanmics -------
    opti.subject_to(r(n_time+1)<=r(n_time)+rho*delta(n_time)-mu);
    
    % ======== uncertainty dynamics ========
    % -------- process noise dynamics --------
    temp_G = G(:,(n_time-1)*order+1:n_time*order);
    dG = dyn_funda(A,temp_G,order);
    opti.subject_to([G_init,temp_G]*coeff_diff == dG*delta(n_time));
    temp_lambda_f = lambda_f(:,(n_time-1)*order+1:n_time*order);
    dPw = dyn_int(temp_G,R_w,B_w,temp_lambda_f,order); 
    Pw_end = dPw*coeff_quad*delta(n_time)*(temp_lambda_f*coeff_quad*delta(n_time));
    if n_time == 1
        % no feedback in the first interval
        opti.subject_to(P(:,n_time)== Pw_end);
        % state constraint
        temp = mat(P(:,n_time));
        temp = diag(temp);
        opti.subject_to(x(:,n_time*(order+1)+1)+sqrt(sqrt(temp.^2)+1e-4)<=x_max);
        opti.subject_to(x(:,n_time*(order+1)+1)-sqrt(sqrt(temp.^2)+1e-4)>=x_min);
%         % another formulation
%         opti.subject_to(temp<=(x_max-x(:,n_time*(order+1)+1)).^2);
%         opti.subject_to(x(:,n_time*(order+1)+1)<=x_max);
%         opti.subject_to((x(:,n_time*(order+1)+1)-x_min).^2>=temp);
%         opti.subject_to(x(:,n_time*(order+1)+1)>=x_min);

        % input constraint
        opti.subject_to(u(:,n_time)<=u_max);
        opti.subject_to(u_min<=u(:,n_time));
        
        dada = dPw;
    else
        % dynamics from feedback
        temp_G_fb = G_fb(:,(n_time-2)*order+1:(n_time-1)*order);
        dG_fb = dyn_funda_fb(A,temp_G_fb,B_u,K,order);
        opti.subject_to([G_init,temp_G_fb]*coeff_diff==dG_fb*delta(n_time));
        G_end = mat([G_init,temp_G_fb]*coeff_cont);
        Pfb_end = G_end*mat(P(:,n_time-1))*G_end';
        % summation of the ellipsoid from feedback and open loop
        temp = mat(P(:,n_time));
        opti.subject_to(temp==Pfb_end/lambda_cont(n_time-1)+mat(Pw_end)/(1-lambda_cont(n_time-1)));
        % state constriant 
%         % tighter reformulation
%         temp = diag(temp);
%         opti.subject_to(temp<=(x_max-x(:,n_time*(order+1)+1)).^2);
%         opti.subject_to(x(:,n_time*(order+1)+1)<=x_max);
%         opti.subject_to((x(:,n_time*(order+1)+1)-x_min).^2>=temp);
%         opti.subject_to(x(:,n_time*(order+1)+1)>=x_min);
        % sharper version
        diag_P = diag(Pfb_end);
        diag_Pw = diag(mat(Pw_end));
        % upper/lower bound (add 1e-4 to ensure good numerical stability)
        opti.subject_to(x(:,n_time*(order+1)+1)+sqrt(sqrt(diag_Pw.^2)+1e-4)+sqrt(sqrt(diag_P.^2)+1e-4)<=x_max);
        opti.subject_to(x(:,n_time*(order+1)+1)-sqrt(sqrt(diag_Pw.^2)+1e-4)-sqrt(sqrt(diag_P.^2)+1e-4)>=x_min);
        
        % input constraints
        temp = mat(P(:,n_time-1));
        temp = diag(K*temp*K');
%         opti.subject_to(u(n_time)+sqrt(sqrt(temp.^2)+1e-4)<=u_max);
%         opti.subject_to(u(n_time)-sqrt(sqrt(temp.^2)+1e-4)<=u_min);
        % another formulation
        opti.subject_to(temp<=(u_max-u(n_time)).^2);
        opti.subject_to(temp<=(u(n_time)-u_min).^2);
        opti.subject_to(u(:,n_time)<=u_max);
        opti.subject_to(u_min<=u(:,n_time));
        
    end
    
end

% Non-robust constraints
opti.subject_to(delta_min<=delta<=delta_max);
opti.subject_to(r_min<=r<=r_max);

% ---- Setup solver NLP    ------
opti.minimize(cost(end));
ops = struct;
ops.ipopt.print_level = 0;
ops.ipopt.max_iter = 2e4;
ops.ipopt.tol = 1e-4;
opti.solver('ipopt', ops);

%% closed-loop simulation
sim = struct;
sim.x = zeros(dim_x,1); sim.t = 0;
sim.u = []; sim.ref = 1; sim.r = r_max;


while sim.t(end)<22
    fprintf("current time: %.2f\n", sim.t(end));
    % set initial condition
    opti.set_value(x0,sim.x(:,end)); opti.set_value(r0,sim.r(:,end));
    opti.set_value(ref,sim.ref(:,end));
    
    % solve optimization problem
    sol = opti.solve();
    u_temp = sol.value(u(:,1)); delta_temp = sol.value(delta(1));
    r_temp = min(sim.r(end)+rho*delta_temp-mu,r_max);
    w_temp = (w_max-w_min)*rand(dim_w,order+1)+w_min;
    temp = [0,tau,1]*delta_temp;
    x_temp = sim.x(:,end);
    for i = 1:size(w_temp,2)
        [~,x_temp] = ode45(@(t,x) A*x+B_u*u_temp+B_w*w_temp(:,i),[0,temp(i+1)-temp(i)],x_temp);
        x_temp = x_temp(end,:)';
    end
    
    % logging
    sim.x(:,end+1) = x_temp;     % mean + variance
    sim.u(:,end+1) = u_temp; sim.r(end+1) = r_temp; 
    sim.t(end+1) = sim.t(end)+delta_temp;
    sim.ref(end+1) = 0.6*sign(sin(0.4*sim.t(end)))+0.4;
    
    % warm start
    opti.set_initial(u,sol.value([u(:,2:end),u(:,end)]));
    opti.set_initial(x,sol.value([x(:,2:end),x(:,end)]));
    opti.set_initial(delta,delta_max*ones(1,horizon));
    opti.set_initial(r,sol.value([r(:,2:end),r(:,end)]));
    opti.set_initial(cost,sol.value([cost(:,2:end),cost(:,end)]));
    opti.set_initial(P,sol.value(P));
    opti.set_initial(G_fb,sol.value(G_fb));
    opti.set_initial(G,sol.value(G));
%     opti.set_initial(lambda_f_sqrt,sol.value(lambda_f_sqrt));
    
end

%% ploting
figure(1);clf; hold on
stairs(sim.t(1:end),sim.r,'o-');
title('Resource budget');

figure(2);clf; hold on;

stairs(sim.t,sim.ref,'k-');
plot(sim.t,x_max(1)*ones(1,length(sim.t)),'r--','LineWidth',1.5)
plot(sim.t,x_min(1)*ones(1,length(sim.t)),'r--','LineWidth',1.5)
plot(sim.t,sim.x(1,:),'o-');

xlabel('time (s)');
legend('reference','output')
% stairs(sim.t(1:end-1),sim.u,'s-');


figure(3);clf;hold on;
plot(sim.t(1:end-1),diff(sim.t),'o');
title('Event times')
%% helper function
function dG = dyn_funda_fb(A,G,B,K,order)
% fundamental solution of linear feedback system
    dG = [];
    for i = 1:order
        temp = mat(G(:,i));
        temp = A*temp+B*K;
        dG = [dG,vec(temp)];
    end
end

function dG = dyn_funda(A,G,order)
% fundamental solution of linear system
    dG = [];
    for i = 1:order
        temp = mat(G(:,i));
        temp = A*temp;
        dG = [dG,vec(temp)];
    end
end

function dP = dyn_int(G,R_w,B_w,lambda,order)
    % evaluation of the process noise dynamics on the quadrature
    % without the weighting
    dP = [];
    for i = 1:order
        % use the symmetric property of the collocation points
        temp = mat(G(:,order-i+1));
        temp = temp*B_w*R_w*B_w'*temp';
        dP = [dP,vec(temp)/lambda(i)];
    end
end

