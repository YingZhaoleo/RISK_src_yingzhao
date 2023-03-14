function [w_pred, w_real] = get_disturbance(day_tot, T_day,w_min,w_max)
    % Produce disturbances
    % day_tot: total days
    % T_day: steps in 1 day
    % var_y: std of measurement noise

    % w_pred: prediction (outdoor temperature, solar radiation)
    % w_real: real w

    w1_rand = w_min + (w_max-w_min)*rand(1,T_day*day_tot);
    w2_rand = w_min + (w_max-w_min)*rand(1,T_day*day_tot);

    w1_pred = zeros(1,T_day*day_tot);
    w2_pred = zeros(1,T_day*day_tot);

    % Disturbances for each day
    for j = 0:day_tot-1
        a1 = 10 + 4*rand;  a2 = 2 + 4*rand; 
        a3 = 0 + 16*rand;        
        % Outdoor temp
        for i = 1:T_day
            w1_pred(i+j*T_day) = a1 + a2*sin(-0.5*pi+(i)/T_day*2*pi);
        end
        % Radiation
        for i = 1:T_day
            if i<=18
            elseif i<=36
                w2_pred(i+j*T_day) = a3*(i-18)/18;
            elseif i<=54
                w2_pred(i+j*T_day) = a3*(54-i)/18;
            end
        end
    end

    w_pred = [w1_pred; w2_pred];
    w_real = [w1_pred+w1_rand; w2_pred+w2_rand];    
end