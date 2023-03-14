classdef Agent<handle
    % class of self-triggered agent with only box constraints considered 
    properties
        % ====== System Dynamics ======
        % ------ state dynamics ------
        dim_x   % dimension of states
        dim_u   % dimension of input
        dyn     % function handle of the dynamics(dyn(x,u))
        % ------ resource dynamics ------
        rho 	% charging rate of resource
        mu      % transmission cost
        
        % constraints on the system
        r_max, r_min            % constraints on the resources 
        delta_max, delta_min    % constraints on trigger frequency
        x_max, x_min            % box constraints on states
        u_max, u_min            % box constraints on control inputs
        
        % ====== Optimization Configuration(Casadi) ======
        opti    % Casadi Opti stack
        % ------ decision variables ------
        x       % system trajectory
        u       % control inputs
        r       % resour trajectory
        t_trigger   % trigger time stamp sequence(ignore the first one)
        x_global    % state evaluated on the global time grid
        
        % loss function
        loss
        
        % ------ time invariant parameters ------
        horizon     % prediction horizon(default: 10)
        Q,R         % penalty matrix on state and input
        t_global    % global time grid
        
        % collocation method terms
        order   % degree of interpolation polynomial(default: 3)
        p_coeff % Lagrangian polynomial coefficient
        p_der   % coeefficient of the derivativef of polynomial
        p_cont  % coefficients of the continuity
        p_quad  %  quadrature rule
           
        % ------ time varying parameters ------
        x0                          % initial states
        r0                          % initial resource
        delta0_min, delta0_max      % reserved for fix the first interval: sampling interval
        u0_min,u0_max               % reserved for fix the first interval: input
        x0_min,x0_max               % when first interval fixed, then autmatrically 
                                    % set to -inf and inf for numerical robustness
        r0_min,r0_max               % similar to x0_min and x0_max
        ref_x                       % reference signal reserved for the local problem(For stability consideration)
        ref_u                       % reference signal reserved for the local problem
        
        % callable Casadi function for gradient and hessian
        grad                        % includes grad.x, grad.u, grad.r, grad.t
        hess                        % includes hess.x, hess.u, hess.r, hess.t
        
    
    end
        
    methods
        function agent = Agent(varargin)
            
            p = inputParser;
            p.addParameter('dyn',[]);p.addParameter('rho',[]);p.addParameter('mu',[]);
            p.addParameter('order',3,@isinteger);
            p.addParameter('horizon',10,@isinteger);
            p.addParameter('dim_x',[]);p.addParameter('dim_u',[]);
            p.addParameter('x_max',[]);p.addParameter('x_min',[]);
            p.addParameter('u_max',[]);p.addParameter('u_min',[]);
            p.addParameter('r_max',[]);p.addParameter('r_min',[]);
            p.addParameter('delta_max',[]);p.addParameter('delta_min',[]);
            p.addParameter('Q',[]); p.addParameter('R',[]);
            p.addParameter('t_global',[]);
            
            parse(p,varargin{:});
            p = p.Results;
            
            agent.dyn = p.dyn;agent.rho = p.rho;agent.mu = p.mu;
            agent.order = p.order;
            agent.horizon = p.horizon;
            agent.dim_x = p.dim_x;agent.dim_u = p.dim_u;
            agent.x_max = p.x_max;agent.x_min = p.x_min;
            agent.u_max = p.u_max;agent.u_min = p.u_min;
            agent.r_min = p.r_min;agent.r_max = p.r_min;
            agent.delta_max = p.delta_max;agent.delta_min = p.delta_min;
            agent.Q = p.Q; agent.R = p.R;
            agent.t_global = p.t_global;
            
            % ------ Local Self-Triggered MPC ------
            agent.opti = casadi.Opti();
            % decision variables
            agent.x = agent.opti.variable(agent.dim_x,agent.horizon*(agent.order+1)+1);
            agent.u = agent.opti.variable(agent.dim_u,agent.horizon);
            agent.r = agent.opti.variable(1,agent.horizon+1);
            agent.t_trigger = agent.opti.variable(1,agent.horizon);
%             agent.x_global = agent.opti.variable(agent.dim_x,agent.horizon);
            
            % optimization parameters
            agent.x0 = agent.opti.parameter(agent.dim_x,1);
            agent.r0 = agent.opti.parameter(1,1);
            agent.ref_x = agent.opti.parameter(agent.dim_x,1);
            agent.ref_u = agent.opti.parameter(agent.dim_u,1);
            
            % parameters for flexible first interval
            agent.delta0_min = agent.opti.parameter(1);
            agent.delta0_max = agent.opti.parameter(1);
            agent.u0_min = agent.opti.parameter(agent.dim_u,1); 
            agent.u0_max = agent.opti.parameter(agent.dim_u,1);
            agent.x0_min = agent.opti.parameter(agent.dim_x,1);
            agent.x0_max = agent.opti.parameter(agent.dim_x,1);
            agent.r0_min = agent.opti.parameter(1);
            agent.r0_max = agent.opti.parameter(1);
            
            % setup collocation method
            tau_root = [0, casadi.collocation_points(agent.order,'legendre')]; % collocation points
            % Coefficient of the Lagrange polynomial
            agent.p_coeff = zeros(agent.order+1,agent.order+1);
            
            % Coefficients of the collocation equation
            agent.p_der = zeros(agent.order+1,agent.order+1);

            % Coefficients of the continuity equation
            agent.p_cont = zeros(agent.order+1, 1);

            % Coefficients of the quadrature function
            agent.p_quad = zeros(agent.order+1, 1);

            % Construct polynomial basis
            for n_colloc=1:agent.order+1
                % Construct Lagrange polynomials to get the polynomial basis at the collocation point
                temp_coeff = 1;
                for r=1:agent.order+1
                    if r ~= n_colloc
                        temp_coeff = conv(temp_coeff, [1, -tau_root(r)]);
                        temp_coeff = temp_coeff / (tau_root(n_colloc)-tau_root(r));
                    end
                end
                % check the matlab polynomial coefficient structure for
                % better understaning
                agent.p_coeff(n_colloc,:) = temp_coeff; % coefficient of basis at collocation point
                % Evaluate the polynomial at the final time to get the coefficients of the continuity equation
                agent.p_cont(n_colloc) = polyval(temp_coeff, 1.0);

                % Evaluate the time derivative of the basis polynomial at 
                % all collocation points to get the coefficients of the continuity equation
                pder = polyder(temp_coeff);
                for r=1:agent.order+1
                    agent.p_der(n_colloc,r) = polyval(pder, tau_root(r));
                end

                % Evaluate the integral of the polynomial to get the coefficients of the quadrature function
                pint = polyint(temp_coeff);
                agent.p_quad(n_colloc) = polyval(pint, 1.0);
            end
            
            % define the MPC (multiple shooting)
            agent.loss = 0;
            for n_time = 1:agent.horizon
                % ----- set up the parameter of constraints -----
                if n_time == 1
                    agent.opti.subject_to(agent.x(:,1)-agent.x0==0); % initial state
                    agent.opti.subject_to(agent.r(1)==agent.r0);  % initial resource 
                    temp_u_min = agent.u0_min; temp_u_max = agent.u0_max;
                    temp_x_min = agent.x0_min; temp_x_max = agent.x0_max;
                    temp_delta_min = agent.delta0_min; temp_delta_max = agent.delta0_max;
                    temp_r_min = agent.r0_min; temp_r_max = agent.r0_max;
                else    
                    % state and input constraints
                    temp_u_min = agent.u_min; temp_u_max = agent.u_max;
                    temp_x_min = agent.x_min; temp_x_max = agent.x_max;
                    temp_delta_min = agent.delta_min; temp_delta_max = agent.delta_max;
                    temp_r_min = agent.r_min; temp_r_max = agent.r_max;
                end
                
                % ------------------------------------------------
                % ------- loop over collocation points -----------
                % ------------------------------------------------
                
                xk_end = agent.p_cont(1)*agent.x(:,(n_time-1)*(agent.order+1)+1); % reserved for end point multiple shooting
                
                % ------input constraints------
                agent.opti.subject_to(temp_u_min<=agent.u(:,n_time)<=temp_u_max);
                for n_colloc = 1:agent.order
                    % ------ state constraints ------
                    % negelect the initial state at t = 0 and the
                    % constraint at end time of the interval is outside the for loop
                    agent.opti.subject_to(temp_x_min<=agent.x(:,(n_time-1)*(agent.order+1)+n_colloc+1)...
                                            <=temp_x_max);
                    
                    % ----- derivative constraints -----
                    % evaluate derivative from polynomial at n_colloc-th
                    % collocation point
                    
                    % ---- calculate state derivative from polynomial ----
                    % derivative of r+1-th basis polynomial at 
                    % n_colloc+1-th collocation point(not includes 0)
                    xp = agent.p_der(1,n_colloc+1)*agent.x(:,(n_time-1)*(agent.order+1)+1); 
                    for r = 1:agent.order
                        xp = xp+agent.p_der(r+1,n_colloc+1)*agent.x(:,(n_time-1)*(agent.order+1)+1+r);
                    end
                    % ---- calculate state derivative from dynamics ----
                    dx = agent.dyn(agent.x(:,(n_time-1)*(agent.order+1)+n_colloc+1),agent.u(:,n_time));
                    % stage cost
                    stage = (agent.x(:,(n_time-1)*(agent.order+1)+n_colloc+1)-agent.ref_x)'*agent.Q*(agent.x(:,(n_time-1)*(agent.order+1)+n_colloc+1)-agent.ref_x)...
                            + (agent.u(:,n_time)-agent.ref_u)'*agent.R*(agent.u(:,n_time)-agent.ref_u);
                   
                    % p_quad(1) is always 0 for collocation method, hence neglect
                    if n_time == 1
                        agent.opti.subject_to(agent.t_trigger(n_time)*dx == xp);
                        % integrate loss
                        agent.loss = agent.loss + agent.p_quad(n_colloc+1)*stage*agent.t_trigger(n_time);
                    else
                        agent.opti.subject_to((agent.t_trigger(n_time)-agent.t_trigger(n_time-1))*dx==xp);
                        % integrate loss
                        agent.loss = agent.loss + agent.p_quad(n_colloc+1)*stage*(agent.t_trigger(n_time)-agent.t_trigger(n_time-1));
                    end
                    
                    % end point evaluation based on the n_colloc+1-th polynomial
                    xk_end = xk_end + agent.p_cont(1+n_colloc)*agent.x(:,(n_time-1)*(agent.order+1)+1+n_colloc);
                end
                
                % state constraints at end point of the interval
                agent.opti.subject_to(temp_x_min<=agent.x(:,n_time*(agent.order+1)+1)...
                                        <=temp_x_max);
                
                % multiple shooting at the xk_end
                agent.opti.subject_to(agent.x(:,n_time*(agent.order+1)+1)==xk_end);
                
                % resource dynamics and constraints
                agent.opti.subject_to(temp_r_min<=agent.r(n_time+1)<=temp_r_max);
                if n_time == 1
                    agent.opti.subject_to(temp_delta_min<=agent.t_trigger(1)...
                                          <=temp_delta_max);
                    agent.opti.subject_to(agent.r(2)<=agent.r(1)...
                                            +agent.rho*agent.t_trigger(1)-agent.mu);
                else
                    agent.opti.subject_to(temp_delta_min<=agent.t_trigger(n_time)-agent.t_trigger(n_time-1)...
                                        <=temp_delta_max);
                    agent.opti.subject_to(agent.r(n_time+1)<=agent.r(n_time)...
                                          +agent.rho*(agent.t_trigger(n_time)-agent.t_trigger(n_time-1))-agent.mu);
                end
                
                
            end
            
            % constraints reserved for coupling
%             for i = 1:agent.horizon
%                 agent.opti.subject_to(agent.poly_eval(agent.t_trigger,agent.t_global(i))==agent.x_global(:,i));
%             end
            
            % define the optimization problem
            agent.opti.minimize(agent.loss);
            % set up the solver
            opts = struct;
            opts.ipopt.print_level = 2;
            opts.print_time = false;
            opts.ipopt.max_iter = 500;
            opts.ipopt.tol = 1e-1;
            agent.opti.solver('ipopt', opts);
            
            
        end
        
        function y = poly_eval(agent,t_trigger,t)
            % evalution of the state from the collocation trajectory
            % t_trigger comes from opti.eval(agent.t_trigger)
            y = 0;
            
            for i = 1:agent.horizon
                if i == 1
                    y = y+agent.lagrangian((i-1)*(agent.order+1)+1,t/t_trigger(1))...
                            *bump(t,0,t_trigger(1));
                else
                    y = y+agent.lagrangian((i-1)*(agent.order+1)+1,(t-t_trigger(i-1))/(t_trigger(i)-t_trigger(i-1)))...
                            *bump(t,t_trigger(i-1),t_trigger(i));
                end
            end
            
            
        end
        
        function y = lagrangian(agent,ind,t)
            % evaluate the Lagragian function starting from ind to
            % ind+order
            
            y = 0;
            
            % loop over order
            for i = 1:agent.order+1
                % loop over collocation points
                for j = 1:agent.order+1
                    y = y+agent.x(:,ind+j-1)*agent.p_coeff(j,i)*t^(agent.order+1-i);
                end
            end
            
        end
        
        function sol = local_prob(agent,fix,state0,x_init,u_init,t_init,r_init,ref_x,ref_u)
            % solve the local problem
            % Arguments:
            %       fix: Boolean, whether fix the first control interval
            %       state0: struc with current measurements, including:
            %           x0,r0: initial state of states and resource
            %           when fix: r1: the prefix resource c
            
            if fix
                u0_min_temp = state0.u0; u0_max_temp = state0.u0; 
                x0_min_temp = -inf; x0_max_temp = inf;
                delta0_min_temp = state0.delta0; delta0_max_temp = state0.delta0;
                r0_min_temp = state0.r1; r0_max_temp = state0.r1;
            else
                u0_min_temp = agent.u_min; u0_max_temp = agent.u_max;
                x0_min_temp = agent.x_min; x0_max_temp = agent.x_max;
                delta0_min_temp = agent.delta_min; delta0_max_temp = agent.delta_max;
                r0_min_temp = agent.r_min; r0_max_temp = agent.r_max;
            end
            
            agent.opti.set_value(agent.x0, state0.x0);
            agent.opti.set_value(agent.r0, state0.r0);
            agent.opti.set_value(agent.u0_min,u0_min_temp);
            agent.opti.set_value(agent.u0_max,u0_max_temp);
            agent.opti.set_value(agent.x0_min,x0_min_temp);
            agent.opti.set_value(agent.x0_max,x0_max_temp);
            agent.opti.set_value(agent.delta0_min,delta0_min_temp);
            agent.opti.set_value(agent.delta0_max,delta0_max_temp);
            agent.opti.set_value(agent.r0_min,r0_min_temp);
            agent.opti.set_value(agent.r0_max,r0_max_temp);
            agent.opti.set_value(agent.ref_x,ref_x);
            agent.opti.set_value(agent.ref_u,ref_u);
            
            agent.opti.set_initial(agent.x,x_init);
            agent.opti.set_initial(agent.u,u_init);
            agent.opti.set_initial(agent.t_trigger,t_init);
            agent.opti.set_initial(agent.r,r_init);
            
            sol = agent.opti.solve();
        end
        
%         function ala_info= ala_info(agent)
%             % ger information of the gradient and hessian for the ALADIN
%             % iteration
%             ala.grad.x = 
%             
%             
%             
%         end
        
    end
end
        
        
        