% This script implements the bilevel robust DeePC on a multizone building model
% while considering time-invariant model first
clear all; close all; clc;
rng(123)
warning('off', 'MATLAB:class:DestructorError'); ... only for yingzhao's laptop, osqp has some issue here
% Monte carlo simulation
iter = 1;
while iter <= 1
    yalmip('clear');
    fprintf('Monte Carlo simulation, iteration: %d\n',iter)
    if iter >1
        clearvars -except log log_mpc iter
    end
    %% configuration of the problem
    % system dynamics 
    % x1 to x4: room temperature of room 1 to 4
    % x5 to x9: internal wall temperature connecting rooms wall 14 12 24 23 34
    % x10 to x13: wall temperature connecting outdoors from room 1 to room 4
    A = [0.8067, 0 , 0, 0, 0.0145, 0.0145, 0 ,0 , 0, 0.1143, 0,0, 0;
         0, 0.8411, 0, 0, 0,  0.0202, 0.0271, 0.0202, 0, 0, 0.0351,0,0;
         0, 0, 0.8111,0, 0, 0, 0, 0.0202, 0.0202, 0, 0, 0.0707, 0;
         0, 0, 0, 0.8214, 0.0218, 0, 0.0172, 0, 0.0156, 0, 0, 0, 0.0684;
         0.0612, 0, 0, 0.0672, 0.8355, 0.0025, 0.0022, 0 , 0, 0.0021, 0, 0, 0.0028;
         0.0612, 0.0854, 0, 0, 0.0012,0.8355, 0.0022, 0, 0, 0.0021, 0.0032, 0, 0;
         0.0009, 0.0714,0.0012,0.0653, 0.0022,0.0022,0.8635,0.0022,0.0025,0.0001, 0.0023, 0, 0;
         0, 0.0672, 0.0672, 0.0002, 0, 0, 0.0022, 0.8355, 0.0022, 0, 0.0024, 0.0027, 0;
         0, 0.0002, 0.0672, 0.0643, 0, 0, 0.0025, 0.0022, 0.8355, 0, 0.0001, 0.0027,0;
         0.1512, 0, 0, 0, 0.0008, 0.0008, 0, 0, 0, 0.7241, 0.0025, 0, 0.0025;
         0, 0.0507, 0, 0, 0, 0.0018, 0.0015, 0.0018, 0, 0.0028, 0.7411, 0.0018, 0;
         0, 0, 0.0989, 0, 0, 0, 0,  0.0022, 0.0022, 0, 0.0025, 0.7172, 0.0018;
         0, 0, 0, 0.001211, 0.0022, 0, 0.0004, 0, 0.0026, 0, 0, 0.0022, 0.7112];
    B = [0.07, 0.001;
        0.001, 0.05;
        0, 0.072;
        0.001, 0.068;
        0.006, 0.002;
        0.006, 0.002;
        0.001, 0.003;
        0, 0.003;
        0, 0.003;
        0.006, 0.0003;
        0.0005, 0.002;
        0, 0.002;
        0.0003, 0.004];
    E = 1e-3 *[30.2170, 2.3912;
               10.2170, 0.6912;
               20.2170, 1.1912;
               22.2170, 1.9912;
               1.5376, 0.1032;
               1.5376, 0.1032;
               0, 0.1032;
               0.734, 0.1032;
               0.734, 0.1032;
               163.1813, 0.3144;
               50.12, 0.3144;
               76.12, 0.3144;
               103.1813, 0.3144];
    C = [eye(4),zeros(4,9)];
    
    meas_std = 0.05; % standard deviation of the measurement noise (not used in the algorithm)
    
    dim_y = size(C,1); dim_u = size(B,2); dim_w = size(E,2);
    
    
    % configuration of control and process noise 
    w_min = -1.5; w_max = 1.5;  ... bound for noise
    u_min =  0; u_max = 45;      ... bound for the control input
    y_min = 20; y_max = 26;      ... bound for the outputs
    Qy = 10*eye(dim_y); R = 1e-3*eye(dim_u);              ... stage cost
    
    % configuration of the lower level prediction problem
    w_g = 10;   % penalty term of the weight on g
    
    % ------ generate data ------
    day_tot = 6;    ... total days of data (training and validation)
    T_day = 96;     ... number of samples per day  
    [w_pred, w_real] = get_disturbance(day_tot, T_day, w_min,w_max);
    x = [20*ones(9,1);15*ones(4,1)];
    u = [];
    for i = 1:(day_tot-2)*T_day
        temp = [];
        
        if x(1,end) > 22
            temp = 10*rand(1);
        else
            temp = 30 + 10*rand(1);
        end
        
        if mean(x(2:4,end))>23
            temp = [temp;20*rand(1)];
        else
            temp = [temp;35 + 10*rand(1)];
        end
        
        u(:,end+1) = temp;
        x(:,end+1) =A*x(:,end)+B*temp + E*w_real(:,i);
    end
    y = C*x+meas_std*randn(dim_y,size(x,2));
    
    % ------ configuration of the Hankle matrices ------
    depth_init = 5;     ... number of inputs use for initialization, 
    horizon = 8;       ... prediction horizon for MPC
    depth = horizon+depth_init;   ... depth of the Hankel matrix w.r.t u, depth for y is depth+1
    [Hu,Hw,Hy,Hbar] = build_hankel(u,w_real(:,1:(day_tot-2)*T_day),y,depth);
    
    Hy_init = Hy(1:dim_y*(depth_init+1),:); Hy_pred = Hy(dim_y*(depth_init+1)+1:end,:);
    Hu_init = Hu(1:dim_u*depth_init,:); Hu_pred = Hu(dim_u*depth_init+1:end,:);
    Hw_init = Hw(1:dim_w*depth_init,:); Hw_pred = Hw(dim_w*depth_init+1:end,:);
    
    num_col = size(Hy_init,2);
    
    % some matrix used for the MPC formulation
    Q_tilde = kron(eye(horizon),Qy); R_tilde = kron(eye(horizon),R);
    w_g_tilde = w_g*eye(num_col);
    
    % inequality constraints for outputs and inputs F_y*y<=f_y, F_u*u<=f_u
    F_u = kron(eye(horizon),[eye(dim_u);-eye(dim_u)]);
    f_u = kron(ones(horizon,1),[u_max*ones(dim_u,1);-u_min*ones(dim_u,1)]);
    F_y = kron(eye(horizon),[eye(dim_y);-eye(dim_y)]); 
    f_y = kron(ones(horizon,1),[y_max*ones(dim_y,1);-y_min*ones(dim_y,1)]);
    % set of disturbance
    F_w = kron(eye(horizon),[eye(dim_w);-eye(dim_w)]);
    f_w = kron(ones(horizon,1),[w_max*ones(dim_w,1);-w_min*ones(dim_w,1)]);
    
    % declare the controller
    agent = RobustDeePC(dim_u,dim_y,dim_w,horizon,depth_init,Q_tilde,R_tilde,w_g_tilde,...
                        F_u,f_u,F_y,f_y,F_w,f_w);
    
    ctrl = agent.get_control({Hy_init,Hy_pred,Hu_init,Hu_pred,Hw_init,Hw_pred});
    

    try
        %% define the robust DeePC problem
        % generate random disturbances condition
        disp('running data-driven method...')
        sim = struct;
        sim.ref = [20*ones(4,32), 23*ones(4,40), 20*ones(4,24)];
        sim.u = u(:,end-depth_init+1:end);
        sim.w = w_real(:,T_day*(day_tot-2)-depth_init+1:T_day*(day_tot-2));
        sim.x = x(:,end);
        sim.y = y(:,end-depth_init:end);
        
        for t = 1:length(sim.ref)
            % get the information for the controller
            temp_u_init = vec(sim.u(:,end-depth_init+1:end));
            temp_y_init = vec(sim.y(:,end-depth_init:end));
            temp_w_init = vec(sim.w(:,end-depth_init+1:end));
        
            [u_opt,flag] = ctrl({temp_u_init,temp_y_init,temp_w_init,...
                                vec(w_pred(:,T_day*(day_tot-2)+t:T_day*(day_tot-2)+t+horizon-1)),...
                                sim.ref(:,t)});
            if flag ~= 0
                msg = yalmiperror(flag);
                error(msg);
            end
        
            % forward one-step
            sim.w(:,end+1) = w_real(:,T_day*(day_tot-2)+t);
            sim.u(:,end+1) = vec(u_opt(1:dim_u));
            sim.x(:,end+1) = A*sim.x(:,end)+B*sim.u(:,end)+E*sim.w(:,end);
            sim.y(:,end+1) = C*sim.x(:,end)+meas_std*randn(dim_y,1);
        
        end
        sim.wy = sim.y(:,depth_init+1:end) - C*sim.x;
        
        %% standard method from sys id to mpc
        % ------ get iddata for sysid -----
        dat_id = iddata(y(:,1:end-1)',[u;w_real(:,1:(day_tot-2)*T_day)]',1);
        % dat_id = detrend(dat_id);
        dat_val = iddata(sim.y(:,1:end-1)',[sim.u;sim.w]');
        % dat_val = detrend(dat_val);
        Mss = n4sid(dat_id,size(A,1));
        [~,FIT,~]=compare(dat_val,Mss,5);
    %     if min(FIT)<=89
    %         fprintf('bad model, almost sure that will be infeasible, continue to next seed.\n');
    %         continue;
    %     end
        % we see that D is zero
        A_id = Mss.A; B_id = Mss.B(:,1:dim_u); E_id = Mss.B(:,dim_u+1:end); C_id = Mss.C;
        
        % Kalman filter is the best! (Luenberger filter sucks)
        sys_temp = ss(Mss.A,Mss.B,Mss.C,[],-1);
        [~,L_kal,~] = kalman(sys_temp,0,meas_std^2*eye(dim_y));
        % get initial state from the identified model
        x_id = 10*ones(size(A,1),1);
        % run the observer to get initial guess
        for i = 1:length(u)
            x_id = A_id*x_id+B_id*u(:,i)+E_id*w_real(:,i)-L_kal*(C_id*x_id-y(:,i));
        end
        %% ------ standard robust mpc(refer to code on Yalmip tutorial)------
        u_rb_mean = sdpvar(dim_u,horizon,1,'full');
        temp = kron((tril(ones(horizon))-eye(horizon)),ones(dim_u,dim_w));
        K_rb = sdpvar(dim_u*horizon,dim_w*horizon,'full').*temp;
        w_rb = sdpvar(dim_w*horizon,1,'full');
        w_rb_mean = sdpvar(dim_w,horizon,'full');
        x0_rb = sdpvar(size(A_id,1),1);
        ref_rb = sdpvar(dim_y,1);
        u_rb = K_rb*w_rb + vec(u_rb_mean);
        
        y_rb = [];
        x_rb_mean = x0_rb;
        xk = x0_rb;
        obj = 0;
        for k = 1:horizon
            x_rb_mean = [x_rb_mean,A_id*x_rb_mean(:,end)+B_id*u_rb_mean(:,k)+E_id*w_rb_mean(:,k)];
            xk = A_id*xk + B_id*u_rb((k-1)*dim_u+1:k*dim_u)+E_id*(w_rb_mean(:,k)+w_rb((k-1)*dim_w+1:k*dim_w));
            y_rb = [y_rb;C_id*xk];
            obj = (C_id*x_rb_mean(:,end)-ref_rb)'*Qy*(C_id*x_rb_mean(:,end)-ref_rb)...
                    +u_rb_mean(:,k)'*R*u_rb_mean(:,k)+obj;
        end
        F = [y_min<=y_rb <= y_max, u_min <= u_rb <= u_max];
        G = [w_min <= w_rb <= w_max];
        
        opts = sdpsettings('solver','gurobi','verbose',1); 
        ctrl_rb = optimizer([F,G, uncertain(w_rb)],obj,opts,{x0_rb,ref_rb,w_rb_mean},u_rb_mean);
        
        %% closed-loop simulation
        disp('running model based method...')
        sim_rb = struct;
        sim_rb.x = x_id;
        sim_rb.x_real = sim.x(:,1); ... use the real model for simulation
        sim_rb.y = y(:,end);    ...
        sim_rb.ref = sim.ref;
        sim_rb.u = [];
        sim_rb.w = [];
        for t = 1:length(sim_rb.ref)
            [temp_u,flag] = ctrl_rb{sim_rb.x(:,end),sim_rb.ref(:,t),w_pred(:,T_day*(day_tot-2)+t:T_day*(day_tot-2)+t+horizon-1)};
            
            if flag ~= 0
                msg = yalmiperror(flag);
                error(msg); ... try another random seed, model based method is easy to fail
            end
        
            % forward one-step
            sim_rb.w(:,end+1) = w_real(:,T_day*(day_tot-2)+t);
            sim_rb.u(:,end+1) = temp_u(:,1);
            sim_rb.x_real = A*sim_rb.x_real(:,end)+B*sim_rb.u(:,end) +E*sim_rb.w(:,end);
            sim_rb.x(:,end+1) = A_id*sim_rb.x(:,end)+B_id*sim_rb.u(:,end) ...
                                +E_id*sim_rb.w(:,end)-L_kal*(C_id*sim_rb.x(:,end)-sim_rb.y(:,end));
            % ensures that the measurement noise is the same in two experiments
            sim_rb.y(:,end+1) = C*sim_rb.x_real(:,end)+sim.wy(:,t+1); 
        
        end
        % logging for monte carlo
        log{iter} = sim;
        log_mpc{iter} = sim_rb;
        iter = iter +1;
    catch e
        pause(1)
        fprintf('last random trial failed in model-based approach, continue to next iter: %d.\n',iter);
        continue;
    end
end

%%
figure(2); clf; 
for iter = 1:length(log)
    subplot(2,2,1); hold on
    h1 =plot(log{iter}.y(1,depth_init+1:end)','g');
    h2 = plot(log_mpc{iter}.y(1,1:end)','k--');
    h3 = plot(log_mpc{iter}.ref(1,:),'b');
    h4 = plot(y_min*ones(1,length(log_mpc{iter}.y)),'r--');
    plot(y_max*ones(1,length(log_mpc{iter}.y)),'r--');
    legend([h1,h2,h3,h4],'deepc: room 1', 'mpc: room 1','ref','constraints')
    subplot(2,2,2); hold on
    h1 = plot(log{iter}.y(2,depth_init+1:end)','g');
    h2 =plot(log_mpc{iter}.y(2,1:end)','k--');
    h3 =plot(log_mpc{iter}.ref(1,:),'b');
    h4 =plot(y_min*ones(1,length(log_mpc{iter}.y)),'r--');
    plot(y_max*ones(1,length(log_mpc{iter}.y)),'r--');
    legend([h1,h2,h3,h4],'deepc: room 2', 'mpc: room 2','ref','constraints')
    subplot(2,2,3); hold on
    h1 = plot(log{iter}.y(3,depth_init+1:end)','g');
    h2 =plot(log_mpc{iter}.y(3,1:end)','k--');
    h3 =plot(log_mpc{iter}.ref(1,:),'b');
    h4 =plot(y_min*ones(1,length(log_mpc{iter}.y)),'r--');
    plot(y_max*ones(1,length(log_mpc{iter}.y)),'r--');
    legend([h1,h2,h3,h4],'deepc: room 3', 'mpc: room 3','ref','constraints')
    subplot(2,2,4); hold on
    h1 =plot(log{iter}.y(4,depth_init+1:end)','g');
    h2 =plot(log_mpc{iter}.y(4,1:end)','k--');
    h3 = plot(log_mpc{iter}.ref(1,:),'b');
    h4 =plot(y_min*ones(1,length(log_mpc{iter}.y)),'r--');
    plot(y_max*ones(1,length(log_mpc{iter}.y)),'r--');
    legend([h1,h2,h3,h4],'deepc: room 4', 'mpc: room 4','ref','constraints')
end

%%
figure(3); clf; 
for iter = 1:length(log)
    subplot(2,1,1); hold on
    h1 =plot(log{iter}.u(1,depth_init+1:end)','g');
    h2 = plot(log_mpc{iter}.u(1,1:end)','k--');
    h4 = plot(u_min*ones(1,length(log_mpc{iter}.y)),'r--');
    plot(u_max*ones(1,length(log_mpc{iter}.y)),'r--');
    legend([h1,h2,h4],'deepc: input 1', 'mpc: input 1','constraints')
    subplot(2,1,2); hold on
    h1 = plot(log{iter}.u(2,depth_init+1:end)','g');
    h2 =plot(log_mpc{iter}.u(2,1:end)','k--');
    h4 =plot(u_min*ones(1,length(log_mpc{iter}.y)),'r--');
    plot(u_max*ones(1,length(log_mpc{iter}.y)),'r--');
    legend([h1,h2,h4],'deepc: input 2', 'mpc: input 2','constraints')
end


