% This code is used to test the Koopman operator method with linearly
% external inputs injection, this case is test on a case where Koopman
% operator must exists
% system
clear all;
close all;
clc

%% define descrite dynamics
x_dim = 2; % Number of states
u_dim = 1; % Number of control inputs
C_hat = [1 1]; % Output matrix: y = C_hat*x
outputs_dim = size(C_hat,1); % Number of outputs
f_ud = @dyn; % Dynamics, defined at the end of the script

%% collect data
num_runs = 20; sim_length = 100;

n_delay = 2; % Number of previous inputs and outputs used for lifting

% run simulation to generate data
y={};
u={};

% Delay-embedded "state" 
% y_k = [outputs_{k} ; u_{k-1} ; outputs_{k-1} ... u_{k-nd} ; outputs_{k-nd} ];
y_dim = (n_delay+1)*outputs_dim+n_delay*u_dim;
for i = 1:num_runs
    x_temp = rand(x_dim,1)*2-1;
    u_temp = rand(u_dim,sim_length+n_delay+2)-0.5;  % one extra u for estimating D, another extra for one more y
    y_current = C_hat*x_temp;
    % generate the initial states of zeta
    for j = 1:n_delay
        y_current = [u_temp(:,j);y_current];
        x_temp = [x_temp,f_ud(0,x_temp(:,end),u_temp(:,j))];
        y_current = [C_hat*x_temp(:,end);y_current];
    end
    y_temp = y_current;
    for j = n_delay+1:sim_length+n_delay+1  
        x_temp = [x_temp,f_ud(0,x_temp(:,end),u_temp(:,j))];
        y_current = [C_hat*x_temp(:,end);u_temp(:,j);...
                    y_current(1:end-u_dim-outputs_dim,:)];% drop the old element and put the new ones
        y_temp = [y_temp,y_current];
    end
    u{i} = u_temp(:,n_delay+1:end);    % ensure that the time stamp in u is synchronous with y
    y{i} = y_temp(:,1:end);      % left the one previous zeta to do lifting
end


%% construct matrix for subspace system identification
p = 15; % Kalman filtering length in the unifying framework
f = 30; % prediction steps into future
% construct matrices
y_p=[];u_p=[];y_f=[];u_f=[];
for i = 1:num_runs
   for j = p+1:sim_length-f+2
      y_p= [y_p,reshape(y{i}(:,j-p:j-1),p*y_dim,1)];
      y_f = [y_f,reshape(y{i}(:,j:j+f-1),f*y_dim,1)];
      u_p= [u_p,reshape(u{i}(:,j-p:j-1),p*u_dim,1)];
      u_f = [u_f,reshape(u{i}(:,j:j+f-1),f*u_dim,1)];
   end
end
z_p = [y_p;u_p];

%% calculating the Gamma matrix and H matrix(use sequential version)
temp = [];
Gamma_Lz = [];
index_Gamma = p*(y_dim+u_dim);
for i = 1:f
    if i == 1
        temp = y_f((i-1)*y_dim+1:i*y_dim,:)/[z_p;u_f(1:i*u_dim,:)];
        H_fi = temp(:,index_Gamma+1:end);
    else
        residue = y_f((i-1)*y_dim+1:i*y_dim,:)-H_fi*u_f(u_dim+1:i*u_dim,:);
        temp = residue/[z_p;u_f(1:u_dim,:)];
        H_fi = [temp(:,index_Gamma+1:end),H_fi];
    end
    Gamma_Lz = [Gamma_Lz;temp(:,1:index_Gamma)];
end

% get the A and C matrix with the optimal weighting
u_f_ortho = eye(size(u_f,2))-u_f'*inv(u_f*u_f')*u_f;
W_2_square = (z_p*u_f_ortho*z_p');
% Or_temp = y_f*u_f_ortho*z_p'/(z_p*u_f_ortho*z_p');
[U,S,V] = svd(Gamma_Lz*sqrtm(W_2_square));
% the order of the system 
order = 13;
Or = U(:,1:order)*real(sqrtm(S(1:order,1:order)));
C = Or(1:y_dim,:);
A = Or(1:end-y_dim,:)\Or(y_dim+1:end,:);

% estimate B and D with probably my custom method
D = H_fi(:,end-u_dim+1:end);
% block transpose H_fi to achieve a correct order
H_fi_temp = [];
for i = 1:f-1
    H_fi_temp = [H_fi(:,(i-1)*u_dim+1:i*u_dim);H_fi_temp];
end
B = Or(1:(f-1)*y_dim,:)\H_fi_temp;

%% validating the linear model
sys_estimated = ss(A,B,C,D,1);
u_sim = rand(u_dim,150)-0.5;
t_sim = [1:size(u_sim,2)];
outputs_true = [zeros(outputs_dim,1)];
x0_temp = zeros(x_dim,1);
for i = 1:t_sim(end-1)
    x0_temp = f_ud(0,x0_temp,u_sim(:,i));
    outputs_true = [outputs_true,C_hat*x0_temp];
end
[outputs_estimated,t,~] = lsim(sys_estimated,u_sim,t_sim);
figure(1)
clf
plot(t_sim,outputs_estimated(:,1),'b','LineWidth',1.5);
hold on;
plot(t_sim,outputs_true(1,:),'r','LineWidth',2);
legend('prediction','real_outputs')

%% learn the lifting map
% define the matrix from u to outputs
G_f = [];   
for i =1:f
    G_f = [G_f,zeros(size(G_f,1),u_dim);...
          H_fi(:,end-u_dim*i+1:end)];
end
% estimate the initial states
x_k = real(Or\(y_f-G_f*u_f));

%%
% run GP to model the lifting map (we use VFE model)
for i=1:order
    hyp_trained{i} = fitrgp(y_p(end-y_dim+1:end,:)',x_k(i,:)','Basis','none','Optimizer','QuasiNewton',...
                    'verbose',1,'FitMethod','exact','PredictMethod','exact');     % optimise hyperparameters
end

%% validate the whole model
x0_temp = 2*rand(x_dim,1)-1;
y_sim = C_hat*x0_temp;
u_sim = rand(u_dim,n_delay)-0.5;
for i = 1:n_delay
    x0_temp = f_ud(0,x0_temp,u_sim(:,i));
    y_sim = [C_hat*x0_temp;u_sim(:,i);y_sim];
end
% lifting
x_lift = [];
for i = 1:order
    [temp,~,~] = predict(hyp_trained{i},y_sim','Alpha',0.01);
    x_lift = [x_lift;temp];
end

% simulation 
u_sim = rand(u_dim,150);
t_sim = [1:size(u_sim,2)];
outputs_true = [y_sim(1:outputs_dim)];
for i = 1:t_sim(end-1)
    x0_temp = f_ud(0,x0_temp,u_sim(:,i));
    outputs_true = [outputs_true,C_hat*x0_temp];
end
[outputs_estimated,t,~] = lsim(sys_estimated,u_sim,t_sim,x_lift);

figure(2)
clf
plot(t_sim,outputs_estimated(:,1),'b','LineWidth',1.5);
hold on;
plot(t_sim,outputs_true(1,:),'r','LineWidth',2);
legend('prediction','real_outputs')

%% dynamics

function f = dyn( t,x,u )
f = [   0.5*x(1,:)+u(1,:)-0.4*exp(-x(2,:))*x(1,:);
       -0.2*(x(2,:)*u(1,:)-x(1,:).^2)+0.5*x(2,:)*x(1,:)];
end
