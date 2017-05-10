function [stateSequence] = Viterbi(Model, obsSequence)
%VITERBI Runs the Viterbi algorithm
%   Takes as input the HMM and the observation sequence and return the most
%   probable (hidden) state sequence.
%   Model is a struct that contains the HMM. It has to contain:
%       1. nbStates: number of states of the HMM
%       2. nbSymbol: number of possible observation in the sequence
%       3. transition: transition matrix; transition(i,j) = probability of
%          ending in j while having being in i at the previous step
%       4. priors: state priors, probability of being in state i at first step 
%       5. obsProb: observation probabilities; obsProb(i,j) probability of observing j| being in i)
%       6. states: string with the state labels (in a consistent order with
%          the other variables)
%       7. symbols: string with the symbol labels (in a consistent order with
%          the other variables)
%   obSequence is a string, describing the sequence of symbols/chars


%%  Precomputations over observation sequence
T = length(obsSequence);
obsIndex = zeros(T);
for i=1:T
    obsIndex(i) = find(obsSequence(i) == Model.symbols);
end
%%  Initialization
T1 = zeros(Model.nbStates, T);    % Trellis variables
T2 = zeros(Model.nbStates, T);

T1(:,1) = Model.priors.*Model.obsProb(:,obsIndex(1));

%%  Algorithm
for i=2:T
    for j=1:Model.nbStates
        [T1(j,i),T2(j,i)] = max(T1(:,i-1).*Model.transition(:,j).*repmat(Model.obsProb(j,obsIndex(i)),Model.nbStates,1));
    end
end

%%  Computation of state sequence
X = zeros(1,T);
[sequenceProbability, Z] = max(T1(:,T));
X(T) = Z;

for i=T:-1:2
    Z = T2(Z,i);
    X(i-1) = Z;
end

stateSequence='';
for i=1:T
    stateSequence(i) = Model.states(X(i));
end
end
