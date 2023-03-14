classdef RobustDeePC < handle
    properties
        dim_u;  ...input dimension
        dim_y;  ...output dimension
        dim_w;  ...disturbance dimension
        horizon;... prediction horizon
        depth_init; ... initialization depth
        depth;  ... depth of the Hankel matrices
        
        % decision variables of the optimization problem
        g;      ... nominal g
        u0;y0;w0;   ... initial parts
        u_norm; w_norm; ... normial part
        K;        ... feedback control law
        ref;        ... reference signal
%         lambda;     ... dual variables of the robust constraints
        
        % some terms that does not change
        Q_tilde;R_tilde; ... standard ingredients in the loss function
        w_g_tilde;       ... penalty on the g
        F_u;f_u;         ... input constraints
        F_y;f_y;         ... output constraints
        F_w;f_w;         ... disturbance set (this might change)
            
    end
    methods
        function agent = RobustDeePC(dim_u,dim_y,dim_w,horizon,depth_init,...
                               Q_tilde,R_tilde,w_g_tilde,...
                               F_u,f_u,F_y,f_y,F_w,f_w)
            agent.dim_u = dim_u;
            agent.dim_y = dim_y;
            agent.dim_w = dim_w;
            agent.horizon = horizon;
            agent.depth_init = depth_init;
            agent.depth = horizon+depth_init;
            agent.Q_tilde= Q_tilde;
            agent.R_tilde = R_tilde;
            agent.w_g_tilde = w_g_tilde;
            agent.F_u = F_u; agent.f_u = f_u;
            agent.F_y = F_y; agent.f_y = f_y;
            agent.F_w = F_w; agent.f_w = f_w;
        end
        
        function ctrl = get_control(agent,H)
            Hy_init = H{1}; Hy_pred = H{2};
            Hu_init = H{3}; Hu_pred = H{4};
            Hw_init = H{5}; Hw_pred = H{6};
            
            % compact form(check the correspondance in the right hand side of KKT equation)
            H = [Hu_init;Hw_init;Hw_pred;Hu_pred];  
            % clear the yalmip memory
            yalmip('clear');
            % ------- define the optimization problem --------
            % ------ parameters ------
            % reference
            agent.ref = sdpvar(agent.dim_y,1,'full');
            % initial states 
            agent.u0 = sdpvar(agent.dim_u*agent.depth_init,1,'full'); 
            agent.y0 = sdpvar((agent.depth_init+1)*agent.dim_y,1,'full');
            agent.w0 = sdpvar(agent.dim_w*agent.depth_init,1,'full');
            % ------ decision variables  ------
            agent.u_norm = sdpvar(agent.dim_u*agent.horizon,1,'full'); ... nominal control input
            agent.w_norm = sdpvar(agent.dim_w*agent.horizon,1,'full'); ... nominal disturbance
            % feedback control law(block lower triangle)
            temp = kron((tril(ones(agent.horizon))-eye(agent.horizon)),ones(agent.dim_u,agent.dim_w));
            agent.K = sdpvar(agent.dim_u*agent.horizon,agent.dim_w*agent.horizon,'full').*temp;
            % the KKT matrix is
            %[Hy_init'*Hy_init+w_g_tilde , H';H,O], use block matrix inversion to
            %directly calculate g, where we need the inversion of
            %Hy_init'*Hy_init+w_g_tilde, which we use matrix inversion lemma to lower
            %the computational cost to O((dim_y*depth_init)^2), following second temp is this
            %inversion
            temp = inv(agent.w_g_tilde);
            % desired inversion
            temp = temp-temp*Hy_init'*((eye((agent.depth_init+1)*agent.dim_y)...
                +Hy_init*temp*Hy_init')\Hy_init)*temp;
            temp_inv = temp*H'/(H*temp*H');
            % nominal g
            g_norm = (temp-temp*H'*((H*temp*H')\H)*temp)*Hy_init'*agent.y0...
                + temp_inv*[agent.u0;agent.w0;agent.w_norm;agent.u_norm];

            y_norm = Hy_pred*g_norm;


            % constraints
            cons = [];
            % robust constraints, each column corresponds to one robust constraint
            % robust input cons
            lambda_u = sdpvar(size(agent.F_w,1),size(agent.F_u,1),'full');
            cons = [cons;lambda_u'*agent.f_w<=agent.f_u-agent.F_u*agent.u_norm];
            for i = 1:size(lambda_u,2)
                cons = [cons;agent.F_w'*lambda_u(:,i)==(agent.F_u(i,:)*agent.K)'];
                cons = [cons;lambda_u(:,i)>=zeros(size(agent.F_w,1),1)];
            end
            % robust output cons
            % perturbation on y_pred is temp_inv(:,(dim_w+dim_u)*depth_init+1:end)[I;K]*w_pred
            temp_F_y = agent.F_y*Hy_pred*temp_inv(:,(agent.dim_w+agent.dim_u)*agent.depth_init+1:end);
            lambda_y = sdpvar(size(agent.F_w,1),size(temp_F_y,1),'full');
            temp = [eye(agent.dim_w*agent.horizon);agent.K];
            cons = [cons;lambda_y'*agent.f_w<=agent.f_y-agent.F_y*y_norm];
            for i = 1:size(lambda_y,2)
                cons = [cons;agent.F_w'*lambda_y(:,i)==(temp_F_y(i,:)*temp)'];
                cons = [cons;lambda_y(:,i)>=zeros(size(agent.F_w,1),1)];
            end


            ref_temp = vec(kron(ones(agent.horizon,1),agent.ref));
            % optimize nominal performance
            obj = (y_norm-ref_temp)'*agent.Q_tilde*(y_norm-ref_temp)...
                    +agent.u_norm'*agent.R_tilde*agent.u_norm;

            % configure the solver
            opts = sdpsettings('solver','gurobi','verbose',1);
            ctrl = optimizer(cons,obj,opts,{agent.u0,agent.y0,agent.w0,agent.w_norm,agent.ref},...
                                           {agent.u_norm});
        
        end
    end
    
    
end