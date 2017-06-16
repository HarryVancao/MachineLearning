%% HMM Model Decoder 

% This code is based on the 1989 Paper A Tutorial on Hidden Markov Models
% by Lawrence Rabiner. Hence the variable names that were used reflect
% those used in the tutorial. 

%% Hidden Markov Model Parameters
% Transition Matrix
A = [0.8 0.2 ; 0.2 0.8];

% Emission Matrix
B = [0.4 0.1 0.4 0.1 ; 0.1 0.4 0.1 0.4];

% Initial State Probabilities
PI = [0.5 ; 0.5];

% observations
% ACGT
% CGTCAG
% 234213
O = [2 3 4 2 1 3];

% number of states 
N = size(PI, 1); 

% number of observations
T = size(O, 2); 

%% Evaluate + Decode

% forward algorithm
alpha = zeros(T, N); 
alpha(1, :) = PI(:) .* B(:, O(1)); 

for i = 2:T 
    for j = 1:N 
        alpha(i, j) = sum(alpha(i - 1, :) .* A(j, :));
    end     
    alpha(i, :) = alpha(i, :) .* B(:, O(i))';
end 
evaluation = sum(alpha(T, :));

% backward algorithm calculate backwards filter. 
beta = zeros(T, N); 
beta(T, :) = 1;
for i = T - 1:-1:1
    for j = 1:N 
        beta(i, j) = sum(beta(i + 1, :) .* A(j, :) .* B(:, O(i + 1))');
    end 
end 

% posterior: forward-backward algorithm 
gamma = zeros(T, N); 
for i = 1:T 
    gamma(i, :) = alpha(i, :) .* beta(i, :); 
    gamma(i, :) = gamma(i, :) ./ sum(alpha(i, :) .* beta(i, :));
end

% Viterbi
delta = zeros(T, N); 
delta(1, :) = PI(:) .* B(:, O(1));
for i = 2:T 
    for j = 1:N 
        
        delta(i, j) = max(A(:, j)' .* delta(i-1, :)) * B(j, O(i));
    end
end

psi = zeros(T, N); 
for i = 2:T    
    for j = 1:N
        [~,  psi(i, j)] = max(A(:, j)' .* delta(i - 1, :));
    end 
end 

% Backtrack and compute optimal state paths.
Q = zeros(T, 1); 
% initialize from delta filter
[~, Q(T)] = max(delta(T, :));
for i = T - 1:-1:1
    Q(i) = psi(i + 1, Q(i + 1));
end 

% print out results
evaluation
Q 

