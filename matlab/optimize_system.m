

%%
% x_human = [] % read in maker center position
% xd_human = [] % read in marker center veloctiy
% x_exo = [] % read in marker side position
% xd_exo = [] % read in marker side velocity


% get the delta vectors
delta_x = x_exo - x_human  
delta_xd = v_exo - v_human
tf = 1;
%fit polynomals to the delta equations for interpolation
t = 1:length(delta_x)   ;
t2 = linspace(0, tf, length(delta_x));
hold on
plot(t2,delta_x)
plot(t2, delta_xd)
y0 = [x_human(1);v_human(1)];
% my_path = ode45(@(t,y) model(t,y, 0.01,0.01, delta_x, delta_xd), t, y0);
% plot(my_path.y(1,:))

%%
% poly_x = polyfit(t2, transpose(delta_x),15);
% poly_v = polyfit(t2,delta_xd,15);




% set up opt varibles
A = []
b = []
Aeq = []
beq = []
lb = [1000;50];
ub = [8000;500];
x0 = [5000,80]; % Initial guess
K = 8000
B = 100

fnc_obj =  @(x)objective(x, x_human,v_human, delta_x, delta_xd);

% run the optimzation
%%
opts = optimoptions('patternsearch','UseCompletePoll',true);


tic % Time the solution
[gains,fval,exitflag,output] = patternsearch(fnc_obj,x0,...
                                    [],[],[],[],lb,ub,@confun,opts)
toc

options = optimoptions(@fmincon,'Display','iter','Algorithm','active-set');
[gains,fval,exitflag,output]  = fmincon(fnc_obj,x0,A,b,Aeq,beq,lb,ub,@confun,options);


y0 = [x_human(1);v_human(1)];


my_path = ode45(@(t,y) model(t,y, gains(1),gains(2), delta_x, delta_xd), t, y0);
    
plot(my_path.y(1,:))
% model function
%function dydt = model(t,y,K,B, poly_x, poly_v)
% function dydt = model(t,y,K,B, poly_x, poly_v)
%   dydt = zeros(2,1);
%   dydt(1) = y(2);
%   %dydt(2) = K* polyval(poly_x,t) + B*polyval(poly_v,t);
%   dydt(2) = K*delta_x(t) + B*delta_xd(t);
% end


%model function
function dydt = model(t,y,K,B, delta_x, delta_xd)
 
  dydt = zeros(2,1);
  dydt(1) = y(2);
  dydt(2) = K*delta_x(int64(t)) + B*delta_xd(int64(t));
  end

% objective funtion
function f = objective(x, x_human,v_human, delta_x, delta_v)
    % Get the optimization vars
    K = x(1)
    B = x(2)
    
    % set up the initial state of the model 
    x0 = [x_human(1);v_human(1)];
    
    % run the model 
    % sol = ode45(@model,[0 1],x0, K, B, poly_x, poly_v);
    time_space = 1:length(delta_x);
    %sol = ode45(@(t,y) model(t,y, K, B, delta_x, delta_v), [0,1], x0);
    sol = ode45(@(t,y) model(t,y, K, B, delta_x, delta_v), time_space, x0);
    
    % get the RMSE error
    ts1 = timeseries(sol.y(1,:));
    ts2 = timeseries(transpose(x_human));
    
    [ts1 ts2] = synchronize(ts1,ts2,'union');
    
%     
%     poly_human = polyfit(tspane2,x_human,15);
%     poly_path = polyfit(sol.x ,sol.y(1,:),15);
%     
%     path_human = polyval( poly_human, tspane2);
%     path = polyval( poly_path,tspane2);
    
    path = reshape(ts2.Data, [], 1);
    path_human = reshape(ts1.Data, [], 1);

    f = sum(sqrt( (path - path_human).^2))/length(x_human); 
   
end


 function [c, ceq] = confun(x)
     % Nonlinear inequality constraints
     c = [];
     % Nonlinear equality constraints
     ceq = [];
 end

 
